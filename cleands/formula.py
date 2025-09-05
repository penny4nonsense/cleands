"""Formula parsing utilities for building design matrices from strings.

This module provides a lightweight, NumPy/Pandas-friendly parser for model
formulas. It supports:

- Intercept handling via the special column "(intercept)".
- Basic terms (column names), interactions with ":" and products with "*".
- Inclusion/exclusion using "+" and "-" (handled via `make_pretty_minus`).
- Parentheses grouping.
- Powers using "**" (or "^" if `USE_CARET=True`).
- "As-is" expressions via `I(<python expression>)`, evaluated against `data`.
- A curated set of NumPy elementwise functions (see `NUMPY_FUNCS`).
- Special polynomial generators such as "quadratic(x1,x2)", "cubic(...)", etc.

Key entry points:
    - parse(): returns x_vars, y_var, conditionals, and the processed DataFrame.
    - design helpers such as generate_interactions().

Notes:
    - The parser mutates a *copy* of the input DataFrame and returns it.
    - (intercept) is always added as a column with value 1 unless excluded.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, Callable
import re
from itertools import chain, combinations, combinations_with_replacement
from functools import partial

################################################################################
#################################  Constants  ##################################
################################################################################

USE_CARET: bool = False
"""Whether to accept '^' as a power operator (translated to '**')."""

LOGGING: bool = False
"""Enable debug prints from parse steps when True."""

NUMPY_FUNCS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'around': lambda x: np.around(x),
    'rint': lambda x: np.rint(x),
    'fix': lambda x: np.fix(x),
    'floor': lambda x: np.floor(x),
    'ceil': lambda x: np.ceil(x),
    'trunc': lambda x: np.trunc(x),
    'diff': lambda x: np.diff(x),
    'ediff1d': lambda x: np.ediff1d(x),
    'exp': lambda x: np.exp(x),
    'expm1': lambda x: np.expm1(x),
    'exp2': lambda x: np.exp2(x),
    'log': lambda x: np.log(x),
    'log10': lambda x: np.log10(x),
    'log2': lambda x: np.log2(x),
    'log1p': lambda x: np.log1p(x),
    'i0': lambda x: np.i0(x),
    'sinc': lambda x: np.sinc(x),
    'signbit': lambda x: np.signbit(x),
    'spacing': lambda x: np.spacing(x),
    'reciprocal': lambda x: np.reciprocal(x),
    'positive': lambda x: np.positive(x),
    'negative': lambda x: np.negative(x),
    'angle': lambda x: np.angle(x),
    'real': lambda x: np.real(x),
    'imag': lambda x: np.imag(x),
    'conj': lambda x: np.conj(x),
    'conjugate': lambda x: np.conjugate(x),
    'sqrt': lambda x: np.sqrt(x),
    'cbrt': lambda x: np.cbrt(x),
    'square': lambda x: np.square(x),
    'absolute': lambda x: np.absolute(x),
    'fabs': lambda x: np.fabs(x),
    'sign': lambda x: np.sign(x),
    'nan_to_num': lambda x: np.nan_to_num(x),
    'real_if_close': lambda x: np.real_if_close(x)
}

################################################################################
##############################  Helper functions  ##############################
################################################################################

def bind(x: list) -> list:
    """Flatten a list of lists by one level.

    Args:
        x (list): List whose elements are iterables to be chained.

    Returns:
        list: Single flattened list.
    """
    return list(chain(*x))


def unique(x: list) -> list:
    """Return list with original order but unique entries.

    Args:
        x (list): Input list.

    Returns:
        list: De-duplicated list preserving first occurrence order.
    """
    outp: list = []
    for item in x:
        if item not in outp:
            outp.append(item)
    return outp


def split_expression(expression: str, delimiter: str = '+') -> list[str]:
    """Split an expression by a delimiter respecting parentheses depth.

    This avoids splitting inside nested parentheses.

    Args:
        expression (str): Expression string to split.
        delimiter (str): Delimiter character (default: '+').

    Returns:
        list[str]: The top-level fragments.
    """
    # Preprocess the expression to remove spaces for easier processing
    expression = expression.replace(' ', '')

    parts: list[str] = []  # To store the split parts of the expression
    current_part: list[str] = []  # To build the current part of the expression
    depth: int = 0  # Track the depth of parentheses nesting

    for char in expression:
        if char == delimiter and depth == 0:
            # Join the characters of the current part and add it to the parts list
            parts += [''.join(current_part)]
            current_part = []
        else:
            depth += 1 if char == '(' else -1 if char == ')' else 0
            current_part += [char]

    # Add the last part to the parts list, if it's not empty
    if current_part: parts += [''.join(current_part)]
    return parts


def match_parens(expression: str) -> Optional[str]:
    """If `expression` is fully parenthesized, return the inside; else None.

    Args:
        expression (str): Candidate string like "(a+b)".

    Returns:
        Optional[str]: Inner content without outer parentheses if valid; otherwise None.
    """
    if expression[0]!='(' or expression[-1]!=')': return None
    current_depth: int = 0
    depth: list[int] = []
    expression = expression[1:-1]
    for char in expression:
        current_depth += 1 if char == '(' else -1 if char == ')' else 0
        depth += [current_depth]
    if any([item<0 for item in depth]): return None
    return expression


def make_pretty_minus(expression: str) -> str:
    """Normalize '-' to '+-' at top level to simplify inclusion/exclusion logic.

    Example:
        "x - y + z" -> "x+-y+z" (and leading '+' removed if present)

    Args:
        expression (str): Raw expression.

    Returns:
        str: Normalized expression.
    """
    outp: str = '+-'.join(split_expression(expression,'-'))
    outp = outp.replace('++', '+')
    if len(outp)!=0 and outp[0] == '+': outp = outp[1:]
    return outp


def bin(x: list) -> dict:
    """Count occurrences of each element in a list.

    Args:
        x (list): Input list.

    Returns:
        dict: Mapping item -> count.
    """
    return {item:sum([1 for newbie in x if newbie==item]) for item in unique(x)}

################################################################################
#############################  Expression Parsing  #############################
################################################################################

def parse(formula: str, data: pd.DataFrame) -> Tuple[list[str], str, list[str], pd.DataFrame]:
    """Parse a full formula into design metadata and a processed DataFrame.

    Syntax:
        y ~ rhs [| conditionals]

    Where `rhs` can contain:
        - column names
        - interactions with ':' (e.g., a:b)
        - products with '*' (expanded to main effects + interactions unless
          distribution is detected)
        - '+' and '-' to include/exclude terms
        - powers via '**' (or '^' if `USE_CARET=True`)
        - '.' to include all columns
        - special generators like 'quadratic(a,b)', etc.
        - 'I(expr)' to evaluate raw Python/NumPy expressions on columns

    The function:
      - adds '(intercept)' to `data`,
      - parses the left-hand side (y),
      - returns selected x-vars (after inclusion/exclusion),
      - returns optional `conditionals` to be passed downstream.

    Args:
        formula (str): Full formula string.
        data (pd.DataFrame): Source data.

    Returns:
        Tuple[list[str], str, list[str], pd.DataFrame]:
            - x_vars: Ordered unique design column names (includes '(intercept)' unless removed).
            - y_var: Dependent variable name.
            - conditionals: Parsed conditional columns (after inclusion/exclusion).
            - processed: A copy of data with derived columns added, restricted to
              [y_var] + x_vars + conditionals.
    """
    y_var, rhs = formula.split('~')
    if '|' in formula:
        rhs, conditionals = formula.split('|')
    else:
        conditionals = ''
    data = data.copy()
    data['(intercept)'] = 1
    parse_term(y_var, data)
    x_vars_included, x_vars_excluded = parse_expression(rhs, data)
    conditionals_included, conditionals_excluded = parse_expression(conditionals, data)
    x_vars_included = ['(intercept)']+x_vars_included
    x_vars = [item for item in x_vars_included if item not in x_vars_excluded and item != y_var]
    conditionals = [item for item in conditionals_included if item not in conditionals_excluded and item != y_var]
    x_vars = unique(x_vars)
    conditionals = unique(conditionals)
    return x_vars, y_var, conditionals, data[[y_var]+x_vars+conditionals]


def parse_expression(expression: str, data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Parse a right-hand-side-like expression into included and excluded terms.

    This function orchestrates:
      1) normalization (`make_pretty_minus`, removing "np." / "numpy.", caret handling),
      2) attempting `parse_basic`,
      3) falling back to `parse_complex`.

    Args:
        expression (str): RHS-like expression (may be empty).
        data (pd.DataFrame): DataFrame to mutate with derived columns.

    Returns:
        tuple[list[str], list[str]]: (included_terms, excluded_terms)

    Notes:
        - Returns `None` on failure, but callers typically rely on truthiness and
          do not expect `None` in normal flows.
    """
    expression = expression.strip()
    expression = make_pretty_minus(expression)
    expression = expression.replace('np.', '')
    expression = expression.replace('numpy.', '')
    if USE_CARET: expression = expression.replace('^','**')
    if result:=parse_basic(expression, data):
        if LOGGING: print('simple',expression,result[0],result[1])
        return result
    if result:=parse_complex(expression, data):
        if LOGGING: print('complex',expression,result[0],result[1])
        return result
    return None


def parse_basic(expression: str, data: pd.DataFrame) -> Optional[tuple[list[str], list[str]]]:
    """Handle simple cases: literals, parentheses, single terms, all-cols, powers, sums.

    Rules (order matters):
        - Empty string -> ([], [])
        - Parenthesized -> parse inner
        - "1" -> intercept
        - Valid term -> [term]
        - "." -> all current columns
        - "(... )**k" -> expand to interactions up to power k
        - "a+b+..." -> sum of sub-expressions (recursively parsed)
        - "-expr" -> invert included/excluded sets (for minus handling)
        - Special power funcs (e.g., "quadratic(a,b)")

    Args:
        expression (str): Candidate expression.
        data (pd.DataFrame): Data to mutate with derived columns.

    Returns:
        Optional[tuple[list[str], list[str]]]: (included, excluded) or None.
    """
    if expression == '':
        return [], []
    if match := match_parens(expression):
        return parse_expression(match, data)
    if expression == '1':
        return ['(intercept)'], []
    if parse_term(expression, data):
        return [expression], []
    if expression == '.':
        return data.columns, []
    if match := re.match(r'^\((.*)\)\*\*(\d*)$',expression):
        if match_parens(f'({match.group(1)})'):
            power: int = int(match.group(2))
            expression = match.group(1)
            included: list[str]
            excluded: list[str]
            included, excluded = parse_expression(expression, data)
            if excluded:
                raise ValueError(f'Expression {expression} seems to have excluded terms in a power')
            included += generate_interactions(included, power=power, data=data)
            return included, excluded
    if len(terms := split_expression(expression)) > 1:
        included: list[str] = []
        excluded: list[str] = []
        for term in terms:
            if result := parse_expression(term, data):
                included += result[0]
                excluded += result[1]
        return included, excluded
    if expression[0]=='-':
        if not (result := parse_expression(expression[1:], data)):
            raise ValueError(f'Expression {expression} cannot be inverted')
        return result[1], result[0]
    if included := check_special_power_funcs(expression, data):
        return included, []
    return None


def parse_complex(expression: str, data: pd.DataFrame) -> Optional[Tuple[list[str], list[str]]]:
    """Handle products '*' and interactions ':' with distribution/expansion logic.

    Strategy:
        - Try splitting by '*' via `parse_complex_expression_by_splitting_on_string`.
          If distributed=True, the product was distributable and we return terms.
          Otherwise, we add generated interactions.
        - Try ':' similarly; ensure resulting string is a valid interaction.

    Args:
        expression (str): Candidate expression with '*' or ':'.
        data (pd.DataFrame): Data to mutate.

    Returns:
        Optional[Tuple[list[str], list[str]]]: (included, excluded) or None.

    Raises:
        ValueError: If negations are detected in product/interaction contexts or
            invalid interaction strings are produced.
    """
    included: list[str]
    excluded: list[str] = []

    if result := parse_complex_expression_by_splitting_on_string(expression,data):
        if result['error']:
            raise ValueError(f'Expression {expression} found negations in product')
        distributed: bool = result['distributed']
        included = result['terms']
        if distributed:
            return included, excluded
        included += generate_interactions(included, data=data)
        return included, excluded
    if result := parse_complex_expression_by_splitting_on_string(expression, data, delimiter=':'):
        if result['error']:
            raise ValueError(f'Expression {expression} found negations in interaction')
        included = result['terms']
        expression = ':'.join(included)
        included = [expression]
        if is_interaction(expression):
            return included, excluded
        else:
            raise ValueError(f'Expression {expression} cannot parse an interaction')
    return None


def parse_complex_expression_by_splitting_on_string(
    expression: str,
    data: pd.DataFrame,
    delimiter: str = '*'
) -> Optional[Dict[str,Any]]:
    """Split by a delimiter ('*' or ':') and attempt recursive parsing/distribution.

    For '*':
        - If any sub-expression yields multiple included terms (and no excluded),
          attempt distribution across the product.
        - If distribution succeeds, return the distributed terms with
          {'error': False, 'distributed': True, 'terms': [...]}.
        - Otherwise, return the collected simple terms and mark
          {'error': False, 'distributed': False, 'terms': [...]}, leaving the caller
          to generate interactions.

    For ':':
        - Just return the list of terms; the caller will validate/construct the
          final interaction string.

    Args:
        expression (str): Input expression.
        data (pd.DataFrame): Data to mutate during parsing/evaluation.
        delimiter (str): Either '*' or ':'.

    Returns:
        Optional[Dict[str, Any]]: A dictionary with keys:
            - 'error' (bool): True if excluded terms invalidated the operation.
            - 'distributed' (bool): True if product distribution occurred.
            - 'terms' (list[str] | None): Collected raw terms when successful.
    """
    mod_expression: str = expression.replace('**','<SPECIAL_DELIMITER>')
    if len(terms := split_expression(mod_expression, delimiter=delimiter))>1:
        new_terms: list = []
        for i,term in enumerate(terms):
            unmod_term = term.replace('<SPECIAL_DELIMITER>','**')
            if result := parse_basic(unmod_term, data):
                included = result[0]
                excluded = result[1]
                if excluded:
                    return {'error':True, 'distributed':False, 'terms':None}
                if len(included)>1:
                    retp = [terms[:i]+[included[j]]+terms[(i+1):] for j in range(len(included))]
                    retp = [parse_expression(delimiter.join(item).replace('<SPECIAL_DELIMITER>', '**'),data) for item in retp]
                    if bind([item[1] for item in retp]):
                        return {'error':True, 'distributed':False, 'terms':None}
                    return {'error': False, 'distributed': True, 'terms': bind([item[0] for item in retp])}
                elif len(included)==1:
                    new_terms += [unmod_term]
                else:
                    ...
        return {'error': False, 'distributed': False, 'terms': new_terms}
    return None

################################################################################
################################  Term parsing  ################################
################################################################################

def parse_term(term: str, data: pd.DataFrame) -> bool:
    """Parse a single term by attempting NumPy func, interaction, or as-is.

    Order:
        1) in_numpy()
        2) is_interaction()
        3) is_as_is()

    Args:
        term (str): A candidate term string.
        data (pd.DataFrame): Data to be mutated if term is derived.

    Returns:
        bool: True if the term was successfully parsed/applied to `data`.
    """
    term = term.strip()
    if in_numpy(term, data): return True
    if is_interaction(term, data): return True
    if is_as_is(term, data): return True
    return False


def is_interaction(expression: str, data: pd.DataFrame) -> bool:
    """Create interaction column for colon-separated terms.

    Example:
        "a:b:c" -> data["a:b:c"] = data["a"] * data["b"] * data["c"]

    Args:
        expression (str): Interaction expression with ':'.
        data (pd.DataFrame): Data to mutate.

    Returns:
        bool: True if an interaction was created; False otherwise.
    """
    if len(terms:=split_expression(expression, ':'))>1:
        outp = data['(intercept)'].copy()
        for term in terms:
            if not parse_term(term, data):
                return False
            outp *= data[term]
        data[expression] = outp
        return True
    return False


def in_numpy(expression: str, data: pd.DataFrame) -> bool:
    """Evaluate a recognized NumPy unary function or treat as existing column.

    If `expression` matches a key in `NUMPY_FUNCS` in the form "<func>(col)",
    the new column is added as that function applied to `data[col]`. If the
    expression is already a column name, this returns True.

    Args:
        expression (str): Either a column name or "<func>(col)".
        data (pd.DataFrame): Data to mutate.

    Returns:
        bool: True if the expression is a known column or created successfully.
    """
    if expression in data.columns: return True
    for name, function in NUMPY_FUNCS.items():
        pattern = r'^' + re.escape(name) + r'\((.*)\)$'
        if match := re.match(pattern, expression):
            inside: str = match.group(1)
            if inside in data.columns:
                data[expression] = function(data[inside])
                return True
    return False


def is_as_is(expression: str, data: pd.DataFrame) -> bool:
    """Evaluate a raw Python/NumPy expression with `I(...)`.

    Replaces bare column names in the interior with `data['col']` and ensures
    bare NumPy function names are qualified with `np.` if present in NUMPY_FUNCS.

    Example:
        I((x1 + x2)**2) or I(sqrt(x))

    Args:
        expression (str): Expression beginning with 'I'.
        data (pd.DataFrame): Data to evaluate against.

    Returns:
        bool: True if successfully evaluated and assigned; False otherwise.
    """
    if expression[0]!='I': return False
    if match := match_parens(expression[1:]):
        for item in data.columns:
            match = match.replace(item, f'data[\'{item}\']')
        for name in NUMPY_FUNCS:
            if not re.search(r'\.'+name+r'\(', match) and re.search(name+r'\(', match):
                match = match.replace(name,'np.'+name)
        data[expression] = eval(match)
        return True
    return False

################################################################################
#########################  Generation of Interactions  #########################
################################################################################

def generate_interactions(x: list[str], data: pd.DataFrame, power: Optional[int] = None) -> list[str]:
    """Generate all unique interaction terms up to a given order.

    Args:
        x (list[str]): Base term names (already validated/created in `data`).
        data (pd.DataFrame): Data to mutate with interactions.
        power (Optional[int]): Maximum interaction order. Defaults to len(x).

    Returns:
        list[str]: Sorted, unique interaction strings that were generated.

    Raises:
        ValueError: If any generated term fails to create an interaction in `data`.
    """
    x = unique(x.copy())
    if power==None: power = len(x)
    x = unique(bind([unique([":".join(item) for item in combinations(x, r=i)]) for i in range(2, power + 1)]))
    for i, item in enumerate(x):
        terms = split_expression(item, delimiter=":")
        terms = unique(terms)
        terms.sort()
        x[i] = ":".join(terms)
    x = unique(x)
    if any([not is_interaction(item, data) for item in x]):
        raise ValueError(f'Terms {x} failed to generate interactions')
    return x

################################################################################
##########################  Special power functions  ###########################
################################################################################

def special_power(terms: list[str], data: pd.DataFrame, power: int = 1) -> Optional[list[str]]:
    r"""Generate special-power terms (linear, quadratic, etc.) for a set of terms.

    Args:
        terms (list[str]): Base terms to expand.
        data (pd.DataFrame): DataFrame where generated columns will be stored.
        power (int, optional): Maximum power to generate. Defaults to 1.

    Returns:
        list[str]: Names of generated terms stored in ``data``.

    Notes:
        - For a single term ``x``, quadratic produces ``x`` and ``I(x**2)``.
        - For multiple terms, interaction powers like ``I(x*y)`` and ``I(x**2*y)`` may be created.

    """
    outp: list[str] = []
    terms += ['1']
    for pairing in combinations_with_replacement(terms, r=power):
        pairing = [item for item in pairing if item!='1']
        match len(pairing):
            case 0:
                continue
            case 1:
                expression = pairing[0]
            case _:
                pairing.sort()
                mydict = bin(pairing)
                if len(mydict)==1:
                    term = list(mydict.keys())[0]
                    power = mydict[term]
                    expression = f'I({f"({term})**{power}" if len(split_expression(term)) > 1 else f"{term}**{power}" if power != 1 else f"{term}"})'
                else:
                    expression = f'I({"*".join([f"(({key})**{value})" if len(split_expression(key)) > 1 else f"({key}**{value})" if value != 1 else f"{key}" for key, value in bin(pairing).items()])})'

        if parse_term(expression, data):
            outp += [expression]
        else:
            raise ValueError(f'Expression: {expression} could not be parsed as a term')
    return outp


special_power_funcs = {
    'linear': partial(special_power, power=1),
    'quadratic': partial(special_power, power=2),
    'cubic': partial(special_power, power=3),
    'quartic': partial(special_power, power=4),
    'quintic': partial(special_power, power=5),
    'sextic': partial(special_power, power=6),
    'hexic': partial(special_power, power=6),
    'septic': partial(special_power, power=7),
    'octic': partial(special_power, power=8),
    'nonic': partial(special_power, power=9),
    'decic': partial(special_power, power=10),
    'duodecic': partial(special_power, power=12),
    'vigintic': partial(special_power, power=20)
}
"""Registry mapping special polynomial keywords to generator callables."""


def check_special_power_funcs(expression: str, data: pd.DataFrame) -> Optional[list[str]]:
    """Detect and expand special polynomial helpers like 'quadratic(...)'.

    Args:
        expression (str): Expression beginning with a registered keyword.
        data (pd.DataFrame): Data to mutate with generated terms.

    Returns:
        Optional[list[str]]: Generated term list if matched; otherwise None.
    """
    for key,func in special_power_funcs.items():
        if expression.startswith(key):
            if match:=match_parens(expression[len(key):]):
                terms = split_expression(match, delimiter=',')
                result = func(terms=terms, data=data)
                return result
            else:
                return
        else:
            ...
    return
