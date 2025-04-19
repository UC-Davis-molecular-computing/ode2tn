from typing import Any, Iterable
import re
from typing import Callable

from scipy.integrate import OdeSolver
import gpac as gp
import sympy as sp
from scipy.integrate._ivp.ivp import OdeResult  # noqa


def plot_tn(
        odes: dict[sp.Symbol | str, sp.Expr | str | float],
        initial_values: dict[sp.Symbol | str, float],
        gamma: float,
        beta: float,
        t_eval: Iterable[float] | None = None,
        t_span: tuple[float, float] | None = None,
        dependent_symbols: dict[sp.Symbol | str, sp.Expr | str] | None = None,
        figure_size: tuple[float, float] = (10, 3),
        symbols_to_plot: Iterable[sp.Symbol | str] |
                         Iterable[Iterable[sp.Symbol | str]] |
                         str |
                         re.Pattern |
                         Iterable[re.Pattern] |
                         None = None,
        show: bool = False,
        method: str | OdeSolver = 'RK45',
        dense_output: bool = False,
        events: Callable | Iterable[Callable] | None = None,
        vectorized: bool = False,
        return_ode_result: bool = False,
        args: tuple | None = None,
        loc: str | tuple[float, float] = 'best',
        **options,
) -> OdeResult | None:
    """
    Plot transcription network (TN) ODEs and initial values.

    For arguments other than odes, initial_values, gamma, and beta, see the documentation for
    `plot` in the gpac library.

    Args:
        odes: polynomial ODEs,
            dict of sp symbols or strings (representing symbols) to sympy expressions or strings or floats
            (representing RHS of ODEs)
            Raises ValueError if any of the ODEs RHS is not a polynomial
        initial_values: initial values,
            dict of sympy symbols or strings (representing symbols) to floats
        gamma: coefficient of the negative linear term in the transcription network
        beta: additive constant in x_top ODE

    Returns:
        Typically None, but if return_ode_result is True, returns the result of the ODE integration.
        See documentation of `gpac.plot` for details.
    """
    tn_odes, tn_inits, tn_ratios = ode2tn(odes, initial_values, gamma, beta)
    dependent_symbols_tn = dict(dependent_symbols) if dependent_symbols is not None else {}
    dependent_symbols_tn.update(tn_ratios)
    symbols_to_plot = dependent_symbols_tn if symbols_to_plot is None else symbols_to_plot
    return gp.plot(
        odes=tn_odes,
        initial_values=tn_inits,
        t_eval=t_eval,
        t_span=t_span,
        dependent_symbols=dependent_symbols_tn,
        figure_size=figure_size,
        symbols_to_plot=symbols_to_plot,
        show=show,
        method=method,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        return_ode_result=return_ode_result,
        args=args,
        loc=loc,
        **options,
    )


def ode2tn(
        odes: dict[sp.Symbol | str, sp.Expr | str | float],
        initial_values: dict[sp.Symbol | str, float],
        gamma: float,
        beta: float,
) -> tuple[dict[sp.Symbol, sp.Expr], dict[sp.Symbol, float], dict[sp.Symbol, sp.Expr]]:
    """
    Maps polynomial ODEs and and initial values to transcription network (represented by ODEs with positive
    Laurent polynomials and negative linear term) simulating it, as well as initial values.

    Args:
        odes: polynomial ODEs,
            dict of sympy symbols or strings (representing symbols) to sympy expressions or strings or floats
            (representing RHS of ODEs)
            Raises ValueError if any of the ODEs RHS is not a polynomial
        initial_values: initial values,
            dict of sympy symbols or strings (representing symbols) to floats
        gamma: coefficient of the negative linear term in the transcription network
        beta: additive constant in x_top ODE

    Return:
        triple (`tn_odes`, `tn_inits`, `tn_ratios`), where `tn_ratios` is a dict mapping each original symbol ``x``
        in the original ODEs to the sympy.Expr ``x_top / x_bot``.
    """
    # normalize initial values dict to use symbols as keys
    initial_values = {sp.Symbol(symbol) if isinstance(symbol, str) else symbol: value
                      for symbol, value in initial_values.items()}

    # normalize odes dict to use symbols as keys
    odes_symbols = {}
    symbols_found_in_expressions = set()
    for symbol, expr in odes.items():
        if isinstance(symbol, str):
            symbol = sp.symbols(symbol)
        if isinstance(expr, (str, int, float)):
            expr = sp.sympify(expr)
        symbols_found_in_expressions.update(expr.free_symbols)
        odes_symbols[symbol] = expr

    # ensure that all symbols that are keys in `initial_values` are also keys in `odes`
    initial_values_keys = set(initial_values.keys())
    odes_keys = set(odes_symbols.keys())
    diff = initial_values_keys - odes_keys
    if len(diff) > 0:
        raise ValueError(f"\nInitial_values contains symbols that are not in odes: "
                         f"{comma_separated(diff)}"
                         f"\nHere are the symbols of the ODES:                     "
                         f"{comma_separated(odes_keys)}")

    # ensure all symbols in expressions are keys in the odes dict
    symbols_in_expressions_not_in_odes_keys = symbols_found_in_expressions - odes_keys
    if len(symbols_in_expressions_not_in_odes_keys) > 0:
        raise ValueError(f"Found symbols in expressions that are not keys in the odes dict: "
                         f"{symbols_in_expressions_not_in_odes_keys}\n"
                         f"The keys in the odes dict are: {odes_keys}")

    # ensure all odes are polynomials
    for symbol, expr in odes_symbols.items():
        if not expr.is_polynomial():
            raise ValueError(f"ODE for {symbol}' is not a polynomial: {expr}")

    return normalized_ode2tn(odes, initial_values, gamma, beta)


def normalized_ode2tn(
        odes: dict[sp.Symbol, sp.Expr],
        initial_values: dict[sp.Symbol, float],
        gamma: float,
        beta: float,
) -> tuple[dict[sp.Symbol, sp.Expr], dict[sp.Symbol, float], dict[sp.Symbol, sp.Expr]]:
    # Assumes ode2tn has normalized and done error-checking

    sym2pair: dict[sp.Symbol, tuple[sp.Symbol, sp.Symbol]] = {}
    tn_ratios: dict[sp.Symbol, sp.Expr] = {}
    for x in odes.keys():
        # create x_t, x_b for each symbol x
        x_top, x_bot = sp.symbols(f'{x}_t {x}_b')
        sym2pair[x] = (x_top, x_bot)
        tn_ratios[x] = x_top / x_bot

    tn_odes: dict[sp.Symbol, sp.Expr] = {}
    tn_inits: dict[sp.Symbol, float] = {}
    for x, expr in odes.items():
        p_pos, p_neg = split_polynomial(expr)

        # replace sym with sym_top / sym_bot for each original symbol sym
        for sym in odes.keys():
            sym_top, sym_bot = sym2pair[sym]
            p_pos = p_pos.subs(sym, sym_top / sym_bot)
            p_neg = p_neg.subs(sym, sym_top / sym_bot)

        x_top, x_bot = sym2pair[x]
        # tn_odes[x_top] = beta + p_pos * x_bot - gamma * x_top
        # tn_odes[x_bot] = p_neg * x_bot ** 2 / x_top + beta * x_bot / x_top - gamma * x_bot
        tn_odes[x_top] = beta * x_top / x_bot + p_pos * x_bot - gamma * x_top
        tn_odes[x_bot] = beta + p_neg * x_bot ** 2 / x_top - gamma * x_bot
        tn_inits[x_top] = initial_values[x]
        tn_inits[x_bot] = 1

    return tn_odes, tn_inits, tn_ratios


def split_polynomial(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """
    Split a polynomial into two parts:
    p1: monomials with positive coefficients
    p2: monomials with negative coefficients (made positive)

    Args:
        expr: A sympy Expression that is a polynomial

    Returns:
        pair of sympy Expressions (`p1`, `p2`) such that expr = p1 - p2

    Raises:
        ValueError: If `expr` is not a polynomial. Note that the constants (sympy type ``Number``)
        are not considered polynomials by the ``is_polynomial`` method, but we do consider them polynomials
        and do not raise an exception in this case.
    """
    if expr.is_constant():
        if expr < 0:
            return sp.S(0), -expr
        else:
            return expr, sp.S(0)

    # Verify it's a polynomial
    if not expr.is_polynomial():
        raise ValueError(f"Expression {expr} is not a polynomial")

    # Initialize empty expressions for positive and negative parts
    p_pos = sp.S(0)
    p_neg = sp.S(0)

    # Convert to expanded form to make sure all terms are separate
    expanded = sp.expand(expr)

    # For a sum, we can process each term
    if expanded.is_Add:
        for term in expanded.args:
            # Get the coefficient
            if term.is_Mul:
                # For products, find the numeric coefficient
                coeff = next((arg for arg in term.args if arg.is_number), 1)
            else:
                # For non-products (like just x or just a number)
                coeff = 1 if not term.is_number else term

            # Add to the appropriate part based on sign
            if coeff > 0:
                p_pos += term
            else:
                # For negative coefficients, add the negated term to p2
                p_neg += -term
    elif expanded.is_Mul:
        # If it's a single term, just check the sign; is_Mul for things like x*y or -x (represented as -1*x)
        coeff = next((arg for arg in expanded.args if arg.is_number), 1)
        if coeff > 0:
            p_pos = expanded
        else:
            p_neg = -expanded
    elif expanded.is_Atom:
        # since negative terms are technically Mul, i.e., -1*x, if it is an atom then it is positive
        p_pos = expanded
    else:
        # For single constant terms without multiplication, just check the sign;
        # in tests a term like -x is actually represented as -1*x, so that's covered by the above elif,
        # but in case of a negative constant like -2, this is handled here
        if expanded > 0:
            p_pos = expanded
        else:
            p_neg = -expanded

    return p_pos, p_neg


def comma_separated(elts: Iterable[Any]) -> str:
    return ', '.join(str(elt) for elt in elts)


def main():
    x_sp, y_sp = gp.species('x y')
    rxns = [
        x_sp >> x_sp + y_sp,
        (3 * y_sp | 2 * y_sp).k(11).r(16.5),
        (y_sp >> gp.empty).k(6.5),
    ]
    odes = gp.crn_to_odes(rxns)
    for var in odes.keys():
        if var.name == 'x':
            x = var
        if var.name == 'y':
            y = var
    # for v,ode in odes.items():
    #     print(f"{v}' = {ode}")
    inits = {
        x: 2,
        y: 1,
    }
    gamma = 2
    beta = 1

    tn_odes, tn_inits, tn_ratios = ode2tn(odes, inits, gamma, beta)


if __name__ == '__main__':
    main()
