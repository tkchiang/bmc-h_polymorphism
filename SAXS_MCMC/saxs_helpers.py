import os
import numpy as np
from numpy import pi, sqrt
from scipy.special import j1  # Bessel function J1
from scipy.special import spherical_jn

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import emcee



def loadDatFile(filename):
    data = np.loadtxt(filename)
    q = data[:, 0] # q values in (1/Å)
    Iq = data[:, 1] # Iq values
    dIq = data[:, 2] # Error
    return q, Iq, dIq


# Load FoXS-simulated SAXS from atomic model (pre-interpolated to match the data).
# Ensure that this file is located in the same directory.
FoXS_BMCH = "5djb_FoXS.dat"
q, Iq_h, err_Iq_h = loadDatFile(FoXS_BMCH)



###############################################
# ANALYTICAL SCATTERING FUNCTION DEFINITIONS
###############################################
def hollow_cylinder_polydisperse(q, radius, sigma_radius, length,
                                 thickness, delta_rho,
                                 n_poly=50, n_int=50):
    """
    Compute I(q) for a hollow cylinder with Gaussian polydispersity in radius.

    Parameters
    ----------
    q : array
        Scattering vector values (1/Å).
    radius : float
        Mean inner radius of the cylinder (Å).
    sigma_radius : float
        Standard deviation of the radius (Å).
    length : float
        Cylinder length (Å).
    thickness : float
        Shell thickness (Å).
    delta_rho : float
        Scattering contrast (sld_shell - sld_solvent) (1e-6/Å^2).
    n_poly : int
        Number of Gaussian quadrature points for polydispersity.
    n_int : int
        Number of angular quadrature points for orientational averaging.

    Returns
    -------
    Iq : ndarray
        Polydisperse-averaged scattering intensity I(q).
    """
    q = np.atleast_1d(q)

    # Gaussian quadrature over polydispersity (±3σ range)
    nsig = 3
    x = np.linspace(-nsig, nsig, n_poly)  # standard normal grid
    weights = np.exp(-0.5 * x**2)
    weights /= np.sum(weights)            # normalize
    radii = radius + sigma_radius * x     # sample radii

    # Gauss–Legendre quadrature points for orientation averaging
    xi, wi = np.polynomial.legendre.leggauss(n_int)
    xi = 0.5 * (xi + 1.0)   # rescale [-1,1] → [0,1]
    wi = 0.5 * wi

    # Storage for averaged I(q)
    I_cyl = np.zeros_like(q, dtype=float)

    # Loop over sampled radii
    for r, w_poly in zip(radii, weights):
        R_outer = r + thickness
        gamma = r / R_outer
        H = length / 2
        V_shell = pi * (R_outer**2 - r**2) * length

        qmat = q[:, None]
        y = sqrt(1 - xi**2)[None, :]

        a_outer = qmat * R_outer * y
        a_inner = qmat * r * y

        with np.errstate(divide="ignore", invalid="ignore"):
            Lambda_outer = np.where(a_outer != 0, 2*j1(a_outer)/a_outer, 1.0)
            Lambda_inner = np.where(a_inner != 0, 2*j1(a_inner)/a_inner, 1.0)

        psi = (Lambda_outer - gamma**2 * Lambda_inner) / (1 - gamma**2)

        sinc = np.sinc(qmat * H * xi / pi)  # sinc(x) = sin(pi x)/(pi x)

        integrand = psi**2 * sinc**2
        orient_avg = np.sum(integrand * wi, axis=1)

        I_cyl += w_poly * (V_shell * delta_rho**2 * orient_avg)

    return I_cyl


def lamellar_stack_polydisperse(q, d_spacing, sigma_d_spacing, Nlayers, Caille_parameter,
                                thickness, delta_rho,
                                n_poly=50):
    """
    Compute I(q) for a lamellar stack with Gaussian polydispersity in d-spacing.

    Parameters
    ----------
    q : array
        Scattering vector values (1/Å).
    d_spacing : float
        Mean lamellar repeat distance (Å).
    sigma_d_spacing : float
        Standard deviation of repeat distance (Å).
    Nlayers : float
        Average number of lamellae in the stack.
    Caille_parameter : float
        Caille fluctuation parameter.
    thickness : float
        Lamella thickness (Å).
    delta_rho : float
        Scattering contrast (sld_shell - sld_solvent) (1e-6/Å^2).
    n_poly : int
        Number of Gaussian quadrature points for polydispersity.

    Returns
    -------
    Iq : ndarray
        Polydisperse-averaged scattering intensity I(q).
    """
    q = np.atleast_1d(q)
    Nlayers = int(round(Nlayers))  # ensure integer
    gamma_E = 0.5772156649         # Euler–Mascheroni constant

    # Gaussian distribution for d-spacing (±3σ)
    nsig = 3
    x = np.linspace(-nsig, nsig, n_poly)
    weights = np.exp(-0.5 * x**2)
    weights /= np.sum(weights)
    d_vals = d_spacing + sigma_d_spacing * x

    # Initialize intensity accumulator
    I_lam = np.zeros_like(q, dtype=float)

    # Loop over Gaussian-sampled d_spacings
    for d, w in zip(d_vals, weights):
        delta = thickness
        qdelta = q * delta

        # Form factor: P(q)
        with np.errstate(divide='ignore', invalid='ignore'):
            Pq = np.where(q == 0, 0.0,
                          (2 * delta_rho**2 / q**2) * (1 - np.cos(qdelta)))

        # Structure factor: S(q)
        Sq = np.ones_like(q)
        for n in range(1, Nlayers):
            alpha_n = (Caille_parameter / (4 * np.pi**2)) * (np.log(np.pi * n) + gamma_E)
            phase = np.cos(q * d * n)
            damp = np.exp(-2 * q**2 * d**2 * alpha_n)
            Sq += 2 * (1 - n / Nlayers) * phase * damp

        # Final intensity contribution
        with np.errstate(divide='ignore', invalid='ignore'):
            Iq_sample = 2 * np.pi * (Pq * Sq) / (q**2 * delta)
            Iq_sample[~np.isfinite(Iq_sample)] = 0.0

        I_lam += w * Iq_sample

    return I_lam


def vesicle_polydisperse(q, radius, sigma_radius,
                         thickness, delta_rho,
                         volfraction=1.0, n_poly=50):
    """
    Vesicle (hollow sphere) form factor with Gaussian polydispersity 
    in the core radius, evaluated explicitly.

    Parameters
    ----------
    q : array
        Scattering vector values (1/Å).
    radius : float
        Mean core radius of the vesicle (Å).
    sigma_radius : float
        Standard deviation of the core radius (Å).
    thickness : float
        Shell thickness (Å).
    delta_rho : float
        Scattering contrast (sld_shell - sld_solvent) (1e-6/Å^2).
    volfraction : float, optional
        Volume fraction of shell material. Default = 1.0.
    n_poly : int, optional
        Number of Gaussian samples for averaging. Default = 50.

    Returns
    -------
    Iq : ndarray
        Polydisperse-averaged scattering intensity I(q).
    """
    q = np.atleast_1d(q)

    # Gaussian distribution for radius
    nsig = 3
    x = np.linspace(-nsig, nsig, n_poly)
    weights = np.exp(-0.5 * x**2)
    weights /= np.sum(weights)
    radii = radius + sigma_radius * x

    # Initialize
    I_ves = np.zeros_like(q, dtype=float)

    # Loop over sampled radii
    for r, w in zip(radii, weights):
        R_core = r
        R_tot  = r + thickness

        V_core  = (4/3) * np.pi * R_core**3
        V_tot   = (4/3) * np.pi * R_tot**3
        V_shell = V_tot - V_core

        qR_core = q * R_core
        qR_tot  = q * R_tot

        # spherical Bessel j1(x)
        j1_core = np.where(qR_core != 0, spherical_jn(1, qR_core), 1/3)
        j1_tot  = np.where(qR_tot  != 0, spherical_jn(1, qR_tot),  1/3)

        # Amplitude terms
        A_core = -3 * V_core * delta_rho * j1_core / qR_core
        A_tot  =  3 * V_tot  * delta_rho * j1_tot  / qR_tot

        # Normalize by shell volume
        Fq = (A_core + A_tot) / V_shell

        # Weighted intensity contribution
        I_ves += w * (volfraction * (Fq**2))

    return I_ves

def combined_model(q, 
                   radius, sigma_radius, length,
                   d_spacing, sigma_d_spacing, Nlayers, Caille_parameter,
                   radius_vesicle, sigma_radius_vesicle,
                   thickness, delta_rho,
                   scale_cyl, scale_lam, scale_vesicle, scale_h, background,
                   n_poly=25, n_int=50):
    """
    Compute the scattering I(q) from a combination of:
        - Hollow cylinders (polydisperse in radius)
        - Polydisperse lamellar stacks (polydisperse in layer spacing)
        - Vesicles (hollow spheres, polydisperse in core radius)
        - Constant background

    Parameters
    ----------
    q : array
        Scattering vector values (1/Å).
    radius, sigma_radius : float
        Mean and std. dev. of the inner radius of the cylinder.
    length_cyl : float
        Length of the cylinder.
    d_spacing, sigma_d_spacing : float
        Mean and std. dev. of lamellar d-spacing.
    Nlayers : float
        Average number of lamellar layers.
    Caille_parameter : float
        Caille fluctuation parameter.
    radius_vesicle, sigma_radius_vesicle : float
        Vesicle core radius and polydispersity (std. dev.).
    thickness : float
        Shell thickness.
    delta_rho : float
        Scattering contrast (sld_shell - sld_solvent) (1e-6/Å^2).
    scale_cyl, scale_lam, scale_vesicle, scale_h : float
        Multiplicative scaling factors for each component.
    background : float
        Constant background intensity.
    n_poly : int
        Number of polydispersity sampling points.
    n_int : int
        Number of angular quadrature points for cylinder.

    Returns
    -------
    Iq_total : ndarray
        Combined scattering intensity I(q).
    """

    # Polydisperse hollow cylinder
    Iq_cyl = hollow_cylinder_polydisperse(
        q=q,
        radius=radius, sigma_radius=sigma_radius, length=length,
        thickness=thickness, delta_rho=delta_rho,
        n_poly=n_poly, n_int=n_int
    )

    # Polydisperse lamellar stack
    Iq_lam = lamellar_stack_polydisperse(
        q=q,
        d_spacing=d_spacing, sigma_d_spacing=sigma_d_spacing, Nlayers=Nlayers, Caille_parameter=Caille_parameter,
        thickness=thickness, delta_rho=delta_rho,
        n_poly=n_poly
    )

    # Polydisperse vesicle
    Iq_vesicle = vesicle_polydisperse(
        q=q,
        radius=radius_vesicle, sigma_radius=sigma_radius_vesicle,
        thickness=thickness, delta_rho=delta_rho,
        volfraction=1.0, n_poly=n_poly
    )

    # Combine components with scaling and background
    Iq_total = scale_cyl*Iq_cyl + scale_lam*Iq_lam + scale_vesicle*Iq_vesicle + scale_h*Iq_h + background

    return Iq_total



###############################################
# EXTRACT MAP ESTIMATES
###############################################
def getSamplesLogProbThetaMap(sampler, burnin=1, thin=1):
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    log_prob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    theta_map = samples[np.argmax(log_prob)]

    return samples, log_prob, theta_map




###############################################
# SHOW WALKER TRACES
###############################################
def show_walker_traces(reader, burnin=1, thin=1, labs=[]):

    chain_full = reader.get_chain() # (n_steps, n_walkers, ndim)
    ndim = chain_full.shape[2]
    samples = chain_full[burnin:, :, :]
    
    fig, axes = plt.subplots(ndim, 2, figsize=(10, ndim), sharex='col')
    for i in range(ndim):
        # Full chain
        ax0 = axes[i, 0]
        ax0.plot(chain_full[:, :, i], alpha=0.1)
        ax0.set_ylabel(f"{labs[i]}")
        ax0.set_facecolor("0.96")
        ax0.set_axisbelow(True)
        ax0.grid(True, color="white", linewidth=0.8, zorder=0)
        ax0.tick_params(axis="both", length=0, labelsize=7)
        for spine in ax0.spines.values():
            spine.set_visible(False)
        if i == 0:
            ax0.set_title("Before burn-in removal")
    
        # After burn-in
        ax1 = axes[i, 1]
        ax1.plot(samples[:,:,i], alpha=0.1)
        ax1.set_facecolor("0.96")
        ax1.set_axisbelow(True)
        ax1.grid(True, color="white", linewidth=0.8, zorder=0)
        ax1.tick_params(axis="both", length=0, labelsize=7)
        for spine in ax1.spines.values():
            spine.set_visible(False)
        if i == 0:
            ax1.set_title("After burn-in removal")
    
    axes[-1, 0].set_facecolor("0.96")
    axes[-1, 0].set_axisbelow(True)
    axes[-1, 0].grid(True, color="white", linewidth=0.8, zorder=0)
    axes[-1, 0].tick_params(axis="both", length=0, labelsize=7)
    for spine in axes[-1, 0].spines.values():
        spine.set_visible(False)

    axes[-1, 0].set_xlabel("Step Number")
    axes[-1, 1].set_xlabel("Step Number")


    plt.tight_layout()
    plt.show()
    
    return None





###############################################
# SHOW DATA & SCATTERING CONTRIBUTIONS
###############################################
def showIndividualCurves(q, Iq, err_Iq, params, labels):
    """
    Plot individual scattering contributions (cylinder, lamella, vesicle, subunit)
    alongside the combined model and experimental data.
    """
    idx = {label: i for i, label in enumerate(labels)}
    
    # parameter groups
    groups = {
        "cyl": ["radius_cyl", "sigma_radius_cyl", "length", "thickness", "delta_rho"],
        "lam": ["d_spacing", "sigma_d_spacing", "Nlayers", "Caille_parameter", "thickness", "delta_rho"],
        "ves": ["radius_ves", "sigma_radius_ves", "thickness", "delta_rho"],
    }
    
    # extract grouped params
    cyl_params = params[[idx[k] for k in groups["cyl"]]]
    lam_params = params[[idx[k] for k in groups["lam"]]]
    ves_params = params[[idx[k] for k in groups["ves"]]]

    
    # scale/background terms
    scales = {k: params[idx[k]] for k in ["scale_cyl", "scale_lam", "scale_ves", "scale_h"]}
    background = params[idx["background"]]
    
    # contributions
    I_cyl = scales["scale_cyl"] * hollow_cylinder_polydisperse(q, *cyl_params) + background
    I_lam = scales["scale_lam"] * lamellar_stack_polydisperse(q, *lam_params) + background
    I_ves = scales["scale_ves"] * vesicle_polydisperse(q, *ves_params) + background
    I_h   = scales["scale_h"]   * Iq_h + background
    I_combined = combined_model(q, *params)

    # print parameter groups
    print(f"cylinder params: r = {cyl_params[0]:.3g}, σ_r = {cyl_params[1]:.3g}, L = {cyl_params[2]:.3g}")
    print(f"lamella params: d = {lam_params[0]:.3g}, σ_d = {lam_params[1]:.3g}, N = {lam_params[2]:.3g}, η (Caille) = {lam_params[3]:.3g}")
    print(f"vesicle params: r = {ves_params[0]:.3g}, σ_r = {ves_params[1]:.3g}")
    print(f"material params: t = {ves_params[2]:.3g}, Δρ = {ves_params[3]:.3g}")
    print(f"scales: cyl = {scales['scale_cyl']:.3g}, lam = {scales['scale_lam']:.3g}, ves = {scales['scale_ves']:.3g}, h = {scales['scale_h']:.3g}")
    print(f"background: bg = {background:.3g}")

    
    # plotting
    plt.figure(figsize=(10, 5))
    plt.errorbar(q, Iq, yerr=err_Iq, fmt='.', alpha=0.3, label="Data")
    plt.plot(q, I_combined, color='k', linewidth=1, label="Combined")
    
    contributions = [
        (I_cyl, "Hollow cylinder (polydisperse radius)"),
        (I_lam, "Lamella stacks (polydisperse layer spacing)"),
        (I_ves, "Hollow sphere (polydisperse radius)"),
        (I_h,   "BMC-H subunit only"),
    ]
    for I, lbl in contributions:
        plt.plot(q, I, '--', linewidth=1, label=lbl)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("q (1/Å)")
    plt.ylabel("I(q)")
    plt.legend()
    plt.tight_layout()

    plt.grid()
    plt.show()






###############################################
# SHOW COMPREHENSIVE ANALYSIS RESULTS
###############################################
def show_mcmc_analysis(
    data_filename,
    backend_filename,
    labels,
    burnin=1,
    thin=1,
    bins=20,
    figsize=(12, 8),
    # Corner label styling parameters
    label_fontsize=9,
    label_pad=2,
    label_rotation_x=0,
    label_rotation_y=0
):

    '''
    This function shows a comprehensive summary plot of the MCMC analysis.
    Plots of walker traces enables inspection of convergence of the walkers post-burnin.
    Corner plots show joint distributions, enabling inspection of parameter correlations and marginal distributions.
    Finally, the data, inferred fit, and residuals are shown to assess fit quality.
    '''

    # Load experiment data
    data = np.loadtxt(data_filename)
    q, Iq, err_Iq = data[:, 0], data[:, 1], data[:, 2]

    # Load sample chains from backend file
    reader = emcee.backends.HDFBackend(backend_filename)
    chain_full = reader.get_chain()
    flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)
    ndim = flat_samples.shape[1]

    # Compute log_probs and parameters MAP estimates
    log_prob = reader.get_log_prob(discard=burnin, thin=thin, flat=True)
    map_params = flat_samples[np.argmax(log_prob)]

    # Compute MAP fit curve
    I_combined = combined_model(q, *map_params)

    # Compute scattering contributions
    cyl_idx=(0, 1, 2, 9, 10)
    cyl_params = map_params[list(cyl_idx)]
    lam_idx=(3, 4, 5, 6, 9, 10)
    lam_params = map_params[list(lam_idx)]
    ves_idx=(7, 8, 9, 10)
    ves_params = map_params[list(ves_idx)]
    scales_idx=(11, 12, 13, 14)
    scales = map_params[list(scales_idx)]
    bg = map_params[15]

    I_cyl = scales[0] * hollow_cylinder_polydisperse(q, *cyl_params) + bg
    I_lam = scales[1] * lamellar_stack_polydisperse(q, *lam_params) + bg
    I_ves = scales[2] * vesicle_polydisperse(q, *ves_params) + bg
    I_h   = scales[3] * Iq_h + bg





    # Define figure layout
    fig = plt.figure(figsize=figsize)
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3], wspace=0.1, figure=fig)
    right_spec = outer_gs[1]




    #### PANEL A: Walker Traces (Left Column)
    samples_chain = chain_full[burnin::thin, :, :]
    trace_gs = gridspec.GridSpecFromSubplotSpec(ndim, 1, subplot_spec=outer_gs[0], hspace=0.1)

    trace_axes = []
    for i in range(ndim):
        ax = fig.add_subplot(trace_gs[i], sharex=trace_axes[0] if i > 0 else None)
        trace_axes.append(ax)

        ax.plot(samples_chain[:, :, i], color="black", alpha=0.1, lw=0.5)
        ax.set_ylabel(labels[i], fontsize=8)

        ax.set_facecolor("0.96")
        ax.set_axisbelow(True)
        ax.grid(True, color="white", linewidth=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(axis="both", length=0, labelsize=7, labelbottom=(i == ndim - 1))

        if i == 0:
            ax.set_title(f"Walker Traces ({data_filename.split('/')[-1]})", fontsize=10, pad=8)

    trace_axes[-1].set_xlabel("Step Number", fontsize=9)





    #### PANEL B & C: Right Grid setup for Corner Plot & Fit Panel
    right_gs = gridspec.GridSpecFromSubplotSpec(ndim, ndim, subplot_spec=right_spec, hspace=0.1, wspace=0.1)

    lims = []
    for i in range(ndim):
        lo, hi = np.min(flat_samples[:, i]), np.max(flat_samples[:, i])
        pad = 0.05 * (hi - lo)
        lims.append((lo - pad, hi + pad))

    fit_cols_start = 9
    fit_rows_end = 6



    # Populate Corner Grid
    for i in range(ndim):
        for j in range(ndim):
            if i < fit_rows_end and j >= fit_cols_start:
                continue

            if j > i:
                continue

            ax = fig.add_subplot(right_gs[i, j])
            ax.set_facecolor("0.96")
            ax.set_axisbelow(True)
            ax.grid(True, color="white", linewidth=0.8, zorder=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.tick_params(axis="both", length=0, labelsize=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            if i == j:
                data_col = flat_samples[:, i]
                counts, edges = np.histogram(
                    data_col, bins=bins, range=lims[i]
                )
                ax.bar(edges[:-1],counts,width=np.diff(edges),align="edge",
                    color="steelblue",edgecolor="w",lw=0.1,zorder=2)
                ax.set_xlim(lims[i])
                ax.set_yticks([])
            else:
                x, y = flat_samples[:, j], flat_samples[:, i]
                ax.scatter(
                    x,y,s=2,alpha=0.01,color="steelblue",edgecolor="none",
                    rasterized=True,zorder=2)
                ax.set_xlim(lims[j])
                ax.set_ylim(lims[i])

            # Parameter name labels
            if i == ndim - 1:
                ax.set_xlabel(labels[j],fontsize=label_fontsize,
                    labelpad=label_pad,rotation=label_rotation_x,
                    ha="right" if label_rotation_x != 0 else "center",
                )

            if j == 0 and i != 0:
                ax.set_ylabel(labels[i],fontsize=label_fontsize,labelpad=label_pad,rotation=label_rotation_y,va="center")



    #### PANEL C: Data, Fit, and Residual Plots
    fit_gs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=right_gs[0:fit_rows_end, fit_cols_start:ndim],height_ratios=[4, 1],hspace=0.1)
    ax_fit = fig.add_subplot(fit_gs[0])
    ax_res = fig.add_subplot(fit_gs[1], sharex=ax_fit)

    # Plot data
    ax_fit.errorbar(q, Iq, yerr=err_Iq, fmt=".", color="steelblue", alpha=0.3, ms=2, label="Data", zorder=5)

    # Plot I_combined
    ax_fit.plot(q, I_combined, "r-", lw=1, label="Fit", zorder=6, alpha=0.8)

    # Plot morphology contributions
    ax_fit.plot(q, I_cyl, lw=0.5, label='Cyl',   zorder=4, alpha=0.8)
    ax_fit.plot(q, I_lam, lw=0.5, label='Lam',   zorder=3, alpha=0.8)
    ax_fit.plot(q, I_ves, lw=0.5, label='Ves',   zorder=2, alpha=0.8)
    ax_fit.plot(q, I_h,   lw=0.5, label='BMC-H', zorder=1, alpha=0.8)

    ax_fit.set_xscale("log")
    ax_fit.set_yscale("log")
    ax_fit.set_ylabel("I(q)", fontsize=8)
    ax_fit.legend(fontsize=6, loc="upper right", frameon=False)
    ax_fit.set_title("Data and MAP fit", fontsize=9, pad=4)
    ax_fit.set_ylim(bg/10, np.max(Iq)*2)
    ax_fit.set_facecolor("0.96")
    ax_fit.set_axisbelow(True)
    ax_fit.grid(True, color="white", linewidth=0.8, zorder=0)
    ax_fit.tick_params(axis="both", color='white', labelsize=7, labelbottom=False)
    for spine in ax_fit.spines.values():
        spine.set_visible(False)

    # Plot residuals
    residuals = (Iq - I_combined) / err_Iq # Standardized residuals
    ax_res.plot(q, residuals, ".", color="steelblue", ms=2, alpha=0.7)
    ax_res.axhline(0, color="k", ls="--", lw=0.8, zorder=0)
    ax_res.set_xscale("log")
    ax_res.set_xlabel("$q\ (Å^{-1})$", fontsize=8)
    ax_res.set_ylabel("Residuals", fontsize=7)
    ax_res.set_facecolor("0.96")
    ax_res.set_axisbelow(True)
    ax_res.grid(True, color="white", linewidth=0.8, zorder=0)
    ax_res.tick_params(axis="both", length=0, labelsize=7)
    for spine in ax_res.spines.values():
        spine.set_visible(False)

    plt.show()
    return fig