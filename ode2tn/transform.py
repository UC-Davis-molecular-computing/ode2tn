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
        existing_factors: Iterable[sp.Symbol | str] = (),
        verify_negative_term: bool = True,
        t_span: tuple[float, float] | None = None,
        show_factors: bool = False,
        latex_legend: bool = True,
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

        show_factors: if True, then in addition to plotting the ratios x_top/x_bot,
            also plot the factors x_top and x_bot separately in a second plot.
            Mutually exlusive with `symbols_to_plot`, since it is equivalent to setting
            `symbols_to_plot` to ``[ratios, factors]``, where ratios is a list of dependent symbols
            `x=x_top/x_bot`, and factors is a list of symbols with the transcription factors `x_top`, `x_bot`,
            for each original variable `x`.

        latex_legend: If True, surround each symbol name with dollar signs, unless it is already surrounded with them,
            so that the legend is interpreted as LaTeX. If this is True, then the symbol name must either
            start and end with `$`, or neither start nor end with `$`. Unlike in the gpac package, this is True
            by default. The names of transcription factors are automatically surrounded by dollar signs.
            This option makes sure the legend showing original variables (or dependent symbols) also have `$` added
            so as to be interpreted as LaTeX.

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

        existing_factors: iterable of symbols (or strings with names of symbols) to "pass through" unchanged.
            These symbols are not transformed into `x_top` and `x_bot` variables, but are interpreted as
            being transcription factors already. Raises exception if any of these symbols are
            not in the original ODEs. Also raises exception if any have an ode not of the form
            required as a transcriptional network: a Laurent polynomial in the variables with a single negative
            term `-gamma * x`, where `x` is the transription factor.

        verify_negative_term: If False, then do not check that the existing factors
            have an ODE with a single negative term of the form `-gamma*x`. The default is True, but this can lead to
            false positives. If one constructs a sympy expression such as `expr- gamma*x`, it could be that expr
            can be simplyfied to `alpha*x` for some constant `alpha` > `gamma`. Then the expression would be
            simplified to `(alpha-gamma)*x`, so would no longer have a negative term (or if `alpha < gamma`, it would
            have a negative term, but with the wrong coefficient `(alpha-gamma)` instead of `gamma`). However,
            since one writing an ODE by hand would tend to write it already in simplified form (e.g.,
            it would be awkward to write `odes = {x: 1.7*x - 0.5*x}` rather than simply `odes = {x: 1.2*x}`),
            we leave the check on by default. Only turn off this check if you are confident that the
            ODEs already contain a negative term of the form `-gamma*x`.
            It is still verified that the ODE is a Laurent polynomial; since this condition is independent of
            the form of the expression.

    Returns:
        Typically None, but if return_ode_result is True, returns the result of the ODE integration.
        See documentation of `gpac.plot` for details.
    """
    if show_factors and symbols_to_plot is not None:
        raise ValueError("Cannot use both show_factors and symbols_to_plot at the same time.")

    tn_odes, tn_inits, tn_syms = ode2tn(odes, initial_values, gamma=gamma, beta=beta, scale=scale,
                                        existing_factors=existing_factors,
                                        verify_negative_term=verify_negative_term)
    dependent_symbols_tn = dict(dependent_symbols) if dependent_symbols is not None else {}
    tn_ratios = {sym: sym_t/sym_b for sym, (sym_t, sym_b) in tn_syms.items()}
    dependent_symbols_tn.update(tn_ratios)
    symbols_to_plot = dependent_symbols_tn if symbols_to_plot is None else symbols_to_plot

    if show_factors:
        symbols_to_plot = [symbols_to_plot, [factor for pair in tn_syms.values() for factor in pair]]

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
        latex_legend=latex_legend,
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
        ignore: Iterable[sp.Symbol] = (),
        existing_factors: Iterable[sp.Symbol | str] = (),
        verify_negative_term: bool = True,
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

        ignore: variables to ignore and not transform into `x_top` and `x_bot` variables. Note this is different
            from `existing_factors`, in the sense that variables in `ignore` are not put into the output
            ODEs.

        existing_factors: iterable of symbols (or strings with names of symbols) to "pass through" unchanged.
            These symbols are not transformed into `x_top` and `x_bot` variables, but are interpreted as
            being transcription factors already. Raises exception if any of these symbols are
            not in the original ODEs. Also raises exception if any have an ode not of the form
            required as a transcriptional network: a Laurent polynomial in the variables with a single negative
            term `-gamma * x`, where `x` is the transription factor.

        verify_negative_term: If False, then do not check that the existing factors
            have an ODE with a single negative term of the form `-gamma*x`. The default is True, but this can lead to
            false positives. If one constructs a sympy expression such as `expr- gamma*x`, it could be that expr
            can be simplyfied to `alpha*x` for some constant `alpha` > `gamma`. Then the expression would be
            simplified to `(alpha-gamma)*x`, so would no longer have a negative term (or if `alpha < gamma`, it would
            have a negative term, but with the wrong coefficient `(alpha-gamma)` instead of `gamma`). However,
            since one writing an ODE by hand would tend to write it already in simplified form (e.g.,
            it would be awkward to write `odes = {x: 1.7*x - 0.5*x}` rather than simply `odes = {x: 1.2*x}`),
            we leave the check on by default. Only turn off this check if you are confident that the
            ODEs already contain a negative term of the form `-gamma*x`.
            It is still verified that the ODE is a Laurent polynomial; since this condition is independent of
            the form of the expression.

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

    # normalize existing_factors to be symbols
    existing_factors: list[sp.Symbol] = [sp.Symbol(factor) if isinstance(factor, str) else factor
                                         for factor in existing_factors]

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

    return normalized_ode2tn(odes, initial_values, gamma=gamma, beta=beta, scale=scale, ignore=list(ignore),
                             existing_factors=existing_factors, verify_negative_term=verify_negative_term)


def check_x_is_transcription_factor(x: sp.Symbol, ode: sp.Expr, gamma: float, verify_negative_term: bool) -> None:
    """
    Check if 'ode' is a Laurent polynomial with a single negative term -gamma*x.

    Parameters
    ----------
    x
        The symbol that should appear in the negative term.

    ode
        The symbolic expression to check.

    gamma
        The expected coefficient of the negative term.

    verify_negative_term
        Whether to check for the negative term -gamma*x.

    Raises
    ------
    ValueError
        If 'ode' is not a Laurent polynomial or if it doesn't have exactly
        one negative term of the form -gamma*x.
    """
    # Expand the expression to get it in a standard form
    expanded_ode = sp.expand(ode)

    # Check if the expression is a Laurent polynomial
    if not is_laurent_polynomial(expanded_ode):
        raise ValueError(f"The expression `{ode}` is not a Laurent polynomial")

    if not verify_negative_term:
        return

    # Collect terms and check for the negative term -gamma*x
    terms = expanded_ode.as_ordered_terms()

    # Find negative terms that contain x
    negative_terms = [term for term in terms if is_negative(term) and x in term.free_symbols]

    # Check if there's exactly one negative term
    if len(negative_terms) != 1:
        raise ValueError(f"Expected exactly one negative term with {x}, found {len(negative_terms)} in `{ode}`")

    # Check if the negative term has the form -gamma*x
    negative_term = negative_terms[0]

    # Try to extract the coefficient of x
    coeff = sp.Wild('coeff')
    match_dict = negative_term.match(-coeff * x)

    if match_dict is None:
        raise ValueError(f"The negative term {negative_term} doesn't have the form -gamma*{x} in expression {ode}")

    actual_gamma = float(match_dict[coeff])

    # Check if the coefficient is approximately equal to gamma
    if not is_approximately_equal(actual_gamma, gamma):
        raise ValueError(f"Expected coefficient {gamma}, got {actual_gamma} in expression {ode}")


def is_laurent_polynomial(expr: sp.Expr) -> bool:
    """
    Check if an expression is a Laurent polynomial (polynomial with possible negative exponents).

    Parameters
    ----------
    expr
        The symbolic expression to check.

    Returns
    -------
    :
        True if the expression is a Laurent polynomial, False otherwise.
    """
    if expr.is_Add:
        return all(is_laurent_polynomial(term) for term in expr.args)

    if expr.is_Mul:
        return all(is_laurent_monomial(factor) for factor in expr.args)

    # Constants are Laurent polynomials
    if expr.is_number:
        return True

    # Single symbols are Laurent polynomials
    if expr.is_Symbol:
        return True

    return is_laurent_monomial(expr)


def is_laurent_monomial(expr: sp.Expr) -> bool:
    """
    Check if an expression is a Laurent monomial (term with possible negative exponents).

    Parameters
    ----------
    expr
        The symbolic expression to check.

    Returns
    -------
    :
        True if the expression is a Laurent monomial, False otherwise.
    """
    # Handle division expressions (a/b) by rewriting as a*b^(-1)
    if expr.is_Mul and any(arg.is_Pow and arg.args[1] < 0 for arg in expr.args):
        return all(factor.is_Symbol or
                   (factor.is_Pow and factor.args[0].is_Symbol and factor.args[1].is_Integer) or
                   factor.is_number for factor in expr.args)

    # Direct division (x/y) is rewritten to x*y^(-1) by SymPy
    # But we'll handle it explicitly in case it's encountered directly
    if expr.is_Pow:
        base, exp = expr.args
        return base.is_Symbol and exp.is_Integer

    # Handle direct division representation
    if hasattr(expr, 'func') and expr.func == sp.core.mul.Mul and len(expr.args) == 2:
        if hasattr(expr, 'is_commutative') and expr.is_commutative:
            numer, denom = expr.as_numer_denom()
            return numer.is_Symbol and denom.is_Symbol

    return expr.is_Symbol or expr.is_number


def is_approximately_equal(a: float, b: float, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if two numbers are approximately equal within specified tolerances.

    Parameters
    ----------
    a, b
        The numbers to compare.
    rtol
        The relative tolerance.
    atol
        The absolute tolerance.

    Returns
    -------
    :
        True if the numbers are approximately equal, False otherwise.
    """
    return abs(a - b) <= (atol + rtol * abs(b))

def normalized_ode2tn(
        odes: dict[sp.Symbol, sp.Expr],
        initial_values: dict[sp.Symbol, float],
        *,
        gamma: float,
        beta: float,
        scale: float,
        ignore: list[sp.Symbol],
        existing_factors: list[sp.Symbol],
        verify_negative_term: bool,
) -> tuple[dict[sp.Symbol, sp.Expr], dict[sp.Symbol, float], dict[sp.Symbol, tuple[sp.Symbol, sp.Symbol]]]:
    # Assumes ode2tn has normalized and done error-checking

    tn_syms: dict[sp.Symbol, tuple[sp.Symbol, sp.Symbol]] = {}
    for x in odes.keys():
        if x not in existing_factors and x not in ignore:
            # create x_t, x_b for each symbol x
            x_t, x_b = sp.symbols(f'{x}_t {x}_b')
            tn_syms[x] = (x_t, x_b)

    tn_odes: dict[sp.Symbol, sp.Expr] = {}
    tn_inits: dict[sp.Symbol, float] = {}
    for x, ode in odes.items():
        if x in ignore:
            continue
        if x in existing_factors:
            check_x_is_transcription_factor(x, odes[x], gamma=gamma, verify_negative_term=verify_negative_term)
            ode = odes[x]
            for y, (y_t, y_b) in tn_syms.items():
                ratio = y_t / y_b
                ode = ode.subs(y, ratio)
            tn_odes[x] = ode
            tn_inits[x] = initial_values.get(x, 0) * scale
            continue
        p_pos, p_neg = split_polynomial(ode)

        # replace sym with sym_top / sym_bot for each original symbol sym
        for sym in odes.keys():
            if sym in existing_factors or sym in ignore:
                continue
            sym_top, sym_bot = tn_syms[sym]
            ratio = sym_top / sym_bot
            p_pos = p_pos.subs(sym, ratio)
            p_neg = p_neg.subs(sym, ratio)

        x_t, x_b = tn_syms[x]
        tn_odes[x_t] = beta * x_t / x_b + p_pos * x_b - gamma * x_t
        tn_odes[x_b] = beta + p_neg * x_b ** 2 / x_t - gamma * x_b
        tn_inits[x_t] = initial_values.get(x, 0) * scale
        tn_inits[x_b] = scale
        check_x_is_transcription_factor(x_t, tn_odes[x_t], gamma=gamma, verify_negative_term=False)
        check_x_is_transcription_factor(x_b, tn_odes[x_b], gamma=gamma, verify_negative_term=False)

    return tn_odes, tn_inits, tn_syms


def split_polynomial(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """
    Split a polynomial into two parts:
    p1: monomials with positive coefficients
    p2: monomials with negative coefficients (made positive)

    Parameters
    ----------
    expr: A sympy Expression that is a polynomial

    Returns
    -------
    :
        pair of sympy Expressions (`p1`, `p2`) such that expr = p1 - p2

    Raises
    ------
    ValueError:
        If `expr` is not a polynomial. Note that the constants (sympy type ``Number``)
        are not considered polynomials by the ``is_polynomial`` method, but we do consider them polynomials
        and do not raise an exception in this case.
    """
    # if expr.is_constant():
    #     if expr < 0:
    #         return sp.S(0), -expr
    #     else:
    #         return expr, sp.S(0)

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
            if is_negative(term):
                p_neg += -term
            else:
                p_pos += term
    else:
        if is_negative(expanded):
            p_neg = -expanded
        else:
            p_pos = expanded

    return p_pos, p_neg


def comma_separated(elts: Iterable[Any]) -> str:
    return ', '.join(str(elt) for elt in elts)


def is_negative(expr: sp.Symbol | sp.Expr | float | int) -> bool:
    if isinstance(expr, (float, int, sp.Number)):
        return expr < 0
    elif expr.is_Atom or expr.is_Pow:
        return False
    elif expr.is_Add:
        raise ValueError(f"Expression {expr} is not a single term")
    elif expr.is_Mul:
        arg0 = expr.args[0]
        if isinstance(arg0, sp.Symbol):
            return False
        return arg0.is_negative
    else:
        raise ValueError(f"Unrecognized type {type(expr)} of expression `{expr}`")

def main():
    from sympy.abc import p,q,w,z,x

    import gpac as gp
    import numpy as np
    import sympy as sp
    from ode2tn import ode2tn
    import matplotlib.pyplot as plt

    gamma = 20
    beta = 1
    shift = 2  # amount by which to shift oscillator up to maintain positivity
    x, p, q, w, z, zap = sp.symbols('x p q w z zap')
    # objective function f as a sympy expression
    f_exp = sp.exp(-(x - 3) ** 2 / 0.5) + 1.5 * sp.exp(-(x - 5) ** 2 / 1.5)
    f_plot = sp.plot(f_exp, (x, 0, 8), size=(6, 2))

    # next line commented out because it generates a new plot for some reason; uncomment to save the plot to a file
    # f_plot.save("extremum-seek-objective-function.pdf")

    # f as a Python function that can be called with sympy Expression inputs to substitute for the variable x
    def f(expr):
        return f_exp.subs(x, expr)

    omega = 3  # frequency of oscillation
    lmbda = 0.1 * omega
    k = 0.5 * lmbda  # rate of convergence
    a = 0.1  # magnitude of oscillation
    odes = {
        p: omega * (q - shift),
        q: -omega * (p - shift),
        w: -lmbda * (w - shift),  # + f(x)*(p-shift), # we add this after doing the construction
        z: k * (w - shift),
        x: gamma * (z + a * p - x),
    }
    inits = {
        p: 0 + shift,
        q: 1 + shift,
        w: 0 + shift,
        z: 2,  # z(0) sets initial point of search
    }
    t_eval = np.linspace(0, 300, 1000)

    # we manually plot instead of calling gpac.plot in order to show values of x for different initial values of z in one plot

    tn_odes, tn_inits, tn_syms = ode2tn(odes, inits, gamma=gamma, beta=beta, existing_factors=[x])
    wt, wb = tn_syms[w]
    pt, pb = tn_syms[p]
    zt, zb = tn_syms[z]
    tn_odes[wt] += f(x) * (pt / pb - shift) * wb

    x_idx = 0
    found = False
    for var in tn_odes.keys():
        if var == x:
            found = True
            break
        x_idx += 1
    assert found

    plt.figure(figsize=(12, 5))

    for z_init in np.arange(2, 7.5, 0.5):
        tn_inits[zt] = z_init
        tn_inits[zb] = 1
        sol = gp.integrate_odes(tn_odes, tn_inits, t_eval)
        x_vals = sol.y[x_idx]
        label = f"x for z(0)={z_init}"
        times = sol.t
        plt.plot(times, x_vals, label=label, color="blue")

    plt.yticks(range(8))
    plt.ylim(1, 8)
    plt.xlabel("time")
    plt.ylabel(r"$x$", rotation='horizontal')
    plt.axhline(y=5, color='g', linestyle='--', linewidth=1)
    plt.axhline(y=3.084, color='r', linestyle='--', linewidth=1)



if __name__ == '__main__':
    main()
