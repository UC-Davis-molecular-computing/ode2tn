from typing import Any, Iterable
import re
from typing import Callable

from scipy.integrate import OdeSolver
import gpac as gp
import sympy as sp
from scipy.integrate._ivp.ivp import OdeResult  # noqa


def plot_tn(
        odes: dict[sp.Symbol | str, sp.Expr | str | float],
        initial_values: dict[sp.Symbol | str | gp.Specie, float],
        t_eval: Iterable[float] | None = None,
        *,
        gamma: float,
        beta: float,
        scale: float = 1.0,
        t_span: tuple[float, float] | None = None,
        resets: dict[float, dict[sp.Symbol | str, float]] | None = None,
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
        scale: "scaling factor" for the transcription network ODEs. Each variable `x` is replaced by a pair
            (`x_top`, `x_bot`). The initial `x_bot` value is `scale`, and the initial `x_top` value is
            `x*scale`.
        resets:
            If specified, this is a dict mapping times to "configurations"
            (i.e., dict mapping symbols/str to values).
            The configurations are used to set the values of the symbols manually during the ODE integration
            at specific times.
            Any symbols not appearing as keys in `resets` are left at their current values.
            The keys can either represent the `x_top` or `x_bot` variables whose ratio represents the original variable
            (a key in parameter `odes`), or the original variables themselves.
            If a new `x_top` or `x_bot` variable is used, its value is set directly.
            If an original variable `x` is used, its then the `x_top` and `x_bot` variables are set
            as with transforming `inits` to `tn_inits` in `ode2tn`:
            `x_top` is set to `x*scale`, and `x_bot` is set to `scale`.
            The OdeResult returned (the one returned by `solve_ivp` in scipy) will have two additional fields:
            `reset_times` and `reset_indices`, which are lists of the times and indices in `sol.t`
            corresponding to the times when the resets were applied.
            Raises a ValueError if any time lies outside the integration interval, or if `resets` is empty,
            if a symbol is invalid, or if there are symbols representing both an original variable `x` and one of
            its `x_top` or `x_bot` variables.

    Returns:
        Typically None, but if return_ode_result is True, returns the result of the ODE integration.
        See documentation of `gpac.plot` for details.
    """
    tn_odes, tn_inits, tn_syms = ode2tn(odes, initial_values, gamma=gamma, beta=beta, scale=scale)
    dependent_symbols_tn = dict(dependent_symbols) if dependent_symbols is not None else {}
    tn_ratios = {sym: sym_t/sym_b for sym, (sym_t, sym_b) in tn_syms.items()}
    dependent_symbols_tn.update(tn_ratios)
    symbols_to_plot = dependent_symbols_tn if symbols_to_plot is None else symbols_to_plot

    legend = {}
    for sym, (sym_t, sym_b) in tn_syms.items():
        legend[sym_t] = f'${sym}^\\top$'
        legend[sym_b] = f'${sym}^\\bot$'

    if resets is not None:
        resets = update_resets_with_ratios(odes, resets, tn_odes, tn_syms, scale)

    return gp.plot(
        odes=tn_odes,
        initial_values=tn_inits,
        t_eval=t_eval,
        t_span=t_span,
        dependent_symbols=dependent_symbols_tn,
        resets=resets,
        figure_size=figure_size,
        symbols_to_plot=symbols_to_plot,
        legend=legend,
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


def update_resets_with_ratios(odes, resets, tn_odes, tn_syms, scale: float = 1.0) -> dict[float, dict[sp.Symbol, float]]:
    tn_ratios = {sym: sym_t / sym_b for sym, (sym_t, sym_b) in tn_syms.items()}
    # make copy since we are going to change it
    new_resets = {}
    for time, reset in resets.items():
        new_resets[time] = {}
        for x, val in reset.items():
            new_resets[time][x] = val
    resets = new_resets
    # normalize resets keys and check that variables are valid
    for reset in resets.values():
        for x, val in reset.items():
            if isinstance(x, str):
                del reset[x]
                x = sp.symbols(x)
                reset[x] = val
            if x not in odes.keys() and x not in tn_odes.keys():
                raise ValueError(f"Symbol {x} not found in original variables: {', '.join(odes.keys())},\n"
                                 f"nor found in transcription network variables: {', '.join(tn_odes.keys())}")
        # ensure if original variable x is in resets, then neither x_top nor x_bot are in the resets
        # and substitute x_top and x_bot for x in resets
        for x, ratio in tn_ratios.items():
            # x is an original; so make sure neither x_top nor x_bot are in the reset dict
            if x in reset:
                xt, xb = sp.fraction(ratio)
                if xt in reset:
                    raise ValueError(f'Cannot use "top" variable {xt} in resets '
                                     f'if original variable {x} is also used')
                if xb in reset:
                    raise ValueError(f'Cannot use "bottom" variable {xb} in resets '
                                     f'if original variable {x} is also used')
                reset[xt] = reset[x] * scale
                reset[xb] = scale
                del reset[x]
    return resets


def ode2tn(
        odes: dict[sp.Symbol | str, sp.Expr | str | float],
        initial_values: dict[sp.Symbol | str | gp.Specie, float],
        *,
        gamma: float,
        beta: float,
        scale: float = 1.0,
) -> tuple[dict[sp.Symbol, sp.Expr], dict[sp.Symbol, float], dict[sp.Symbol, tuple[sp.Symbol, sp.Symbol]]]:
    """
    Maps polynomial ODEs and and initial values to transcription network (represented by ODEs with positive
    Laurent polynomials and negative linear term) simulating it, as well as initial values.

    Args:
        odes: polynomial ODEs,
            dict of sympy symbols or strings (representing symbols) to sympy expressions or strings or floats
            (representing RHS of ODEs)
            Raises ValueError if any of the ODEs RHS is not a polynomial
        initial_values: initial values,
            dict of sympy symbols or strings (representing symbols) or gpac.Specie (representing chemical
            species, if the ODEs were derived from `gpac.crn_to_odes`) to floats
        gamma: coefficient of the negative linear term in the transcription network
        beta: additive constant in x_top ODE
        scale: "scaling factor" for the transcription network ODEs. Each variable `x` is replaced by a pair
            (`x_top`, `x_bot`). The initial `x_bot` value is `scale`, and the initial `x_top` value is
            `x*scale`.

    Return:
        triple (`tn_odes`, `tn_inits`, `tn_syms`), where `tn_syms` is a dict mapping each original symbol ``x``
        in the original ODEs to the pair ``(x_top, x_bot)``.
    """
    # normalize initial values dict to use symbols as keys
    initial_values_norm = {}
    for symbol, value in initial_values.items():
        if isinstance(symbol, str):
            symbol = sp.symbols(symbol)
        if isinstance(symbol, gp.Specie):
            symbol = sp.symbols(symbol.name)
        initial_values_norm[symbol] = value
    initial_values = initial_values_norm

    # normalize odes dict to use symbols as keys
    odes_normalized = {}
    symbols_found_in_expressions = set()
    for symbol, expr in odes.items():
        if isinstance(symbol, str):
            symbol = sp.symbols(symbol)
        if isinstance(expr, (str, int, float)):
            expr = sp.sympify(expr)
        symbols_found_in_expressions.update(expr.free_symbols)
        odes_normalized[symbol] = expr
    odes = odes_normalized

    # ensure that all symbols that are keys in `initial_values` are also keys in `odes`
    initial_values_keys = set(initial_values.keys())
    odes_keys = set(odes.keys())
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
    for symbol, expr in odes_normalized.items():
        if not expr.is_polynomial():
            raise ValueError(f"ODE for {symbol}' is not a polynomial: {expr}")

    return normalized_ode2tn(odes, initial_values, gamma=gamma, beta=beta, scale=scale)


def normalized_ode2tn(
        odes: dict[sp.Symbol, sp.Expr],
        initial_values: dict[sp.Symbol, float],
        *,
        gamma: float,
        beta: float,
        scale: float,
) -> tuple[dict[sp.Symbol, sp.Expr], dict[sp.Symbol, float], dict[sp.Symbol, tuple[sp.Symbol, sp.Symbol]]]:
    # Assumes ode2tn has normalized and done error-checking

    tn_syms: dict[sp.Symbol, tuple[sp.Symbol, sp.Symbol]] = {}
    for x in odes.keys():
        # create x_t, x_b for each symbol x
        x_t, x_b = sp.symbols(f'{x}_t {x}_b')
        tn_syms[x] = (x_t, x_b)

    tn_odes: dict[sp.Symbol, sp.Expr] = {}
    tn_inits: dict[sp.Symbol, float] = {}
    for x, expr in odes.items():
        p_pos, p_neg = split_polynomial(expr)

        # replace sym with sym_top / sym_bot for each original symbol sym
        for sym in odes.keys():
            sym_top, sym_bot = tn_syms[sym]
            ratio = sym_top / sym_bot
            p_pos = p_pos.subs(sym, ratio)
            p_neg = p_neg.subs(sym, ratio)

        x_t, x_b = tn_syms[x]
        # tn_odes[x_top] = beta + p_pos * x_bot - gamma * x_top
        # tn_odes[x_bot] = p_neg * x_bot ** 2 / x_top + beta * x_bot / x_top - gamma * x_bot
        tn_odes[x_t] = beta * x_t / x_b + p_pos * x_b - gamma * x_t
        tn_odes[x_b] = beta + p_neg * x_b ** 2 / x_t - gamma * x_b
        tn_inits[x_t] = initial_values.get(x, 0) * scale
        tn_inits[x_b] = scale

    return tn_odes, tn_inits, tn_syms


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
    import gpac as gp
    import numpy as np
    import sympy as sp
    from ode2tn import plot_tn, ode2tn

    P = 1
    I = 1
    D = 1
    val, setpoint, bias, integral, derivative_p, derivative_m, delayed_val = \
        sp.symbols('val setpoint bias integral derivative_p derivative_m delayed_val')
    proportional_term = P * (setpoint - val)
    integral_term = I * (integral - 10)
    # derivative_term = D * (temperature - delayed_temperature)
    c = 1
    odes = {
        val: proportional_term + integral_term,  # + derivative_term,
        integral: setpoint - val,
        delayed_val: c * (val - delayed_val),
        setpoint: 0,
    }
    inits = {
        val: 5,
        setpoint: 7,
        integral: 10,
    }
    t_eval = np.linspace(0, 40, 500)
    figsize = (16, 4)
    resets = {
        10: {val: 9},
        20: {val: 3},
        30: {bias: 2},
    }
    gamma = 1
    beta = 1
    # plot_tn(odes, inits, t_eval, gamma=gamma, beta=beta, resets=resets, figure_size=figsize)
    tn_odes, tn_inits, tn_syms = ode2tn(odes, inits, gamma=gamma, beta=beta)
    val_top = tn_syms[val][0]
    tn_odes[val_top] += bias
    tn_odes[bias] = 0
    gp.plot(tn_odes, tn_inits, t_eval, resets=resets, figure_size=figsize)


if __name__ == '__main__':
    main()
