#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Constants presented in papers
SPACING = array([1., 1., 1.])
ISOLATED_THRESHOLD = -600
DOT_ENHANCED_THRESHOLD = 6
FILTERS_AMOUNT = 6
ISOLATED_MIN_VOLUME = 9
ISOLATED_MAX_VOLUME = 500
JUXTAVASCULAR_MIN_VOLUME = 9
JUXTAPLEURAL_MIN_VALUME = 1


# 

# In[ ]:


MIN_RADIUS = 4
MAX_RADIUS = 16

def get_scales(bottom=MIN_RADIUS, top=MAX_RADIUS, 
               amount=FILTERS_AMOUNT):
    radius = (top / bottom) ** (1. / (amount - 1))
    sigmas = [bottom / 4.]
    for i in range(amount - 1):
        sigmas.append(sigmas[0] * (radius ** i + 1))
    return sigmas


# Enhanced filters based on hessian propreties

# In[ ]:


def hessian(field, coords):
    grad = gradient(field)
    axis = [[0, 1, 2], [1, 2], [2]]
    hess = [gradient(deriv, axis=j) 
            for i, deriv in enumerate(grad) 
            for j in axis[i]]

#   [(0, xx), (1, xy), (2, xz), (3, yy), (4, yz), (5, zz)]
#   x, y, z -> 3, 3, x, y, z -> 3, 3, N

    for j in range(len(hess)):
        hess[j] = hess[j][coords]

    return asarray([[hess[0], hess[1], hess[2]],
                    [hess[1], hess[3], hess[4]],
                    [hess[2], hess[4], hess[5]]])


# In[ ]:


def enhanced_filter(patient, coords, sigma):
    filtered = gaussian_filter(patient, sigma=sigma)
    hess = hessian(filtered, coords=coords)
    hess = [hess[:, :, i] for i in range(hess.shape[-1])]
    with Pool(CPU) as pool:
        eigs = pool.map(linalg.eigvalsh, 
                        hess)

    sigma_sqr = sigma ** 2
    z_dot = [sigma_sqr * (eig_val[2] ** 2) / abs(eig_val[0]) 
             if eig_val[0] < 0 
             and eig_val[1] < 0 
             and eig_val[2] < 0 
             else 0
             for eig_val in eigs]

    z_line = [sigma_sqr * abs(eig_val[1]) 
              * (abs(eig_val[1]) - abs(eig_val[2])) 
              / abs(eig_val[0]) 
              if eig_val[0] < 0 
              and eig_val[1] < 0 
              else 0
              for eig_val in eigs]
    return z_dot, z_line


# In[ ]:


def apply_enhs_filters(patient, mask, include_plane=False):
    sigmas = get_scales()
    enh_dot = zeros(mask.shape)
    enh_line = zeros(mask.shape)
    coords = where(mask)
    
    z_dot = list()
    z_line = list()
    for sigma in sigmas:
        dot, line = enhanced_filter(patient, coords, sigma)
        z_dot.append(dot)
        z_line.append(line)


    enh_dot[coords] = asarray(z_dot).max(axis=0)
    enh_line[coords] = asarray(z_line).max(axis=0)

    return enh_dot, enh_line


# 

# In[ ]:


def div_of_norm_grad(sigma, patient):
    grad = asarray(gradient(patient))
    grad /= norm(grad, axis=0) + 1e-3 # Smooth const
    grad = [gaussian_filter(deriv, sigma=sigma) for deriv in grad]
    return sum([gradient(el, axis=i) 
                for i, el in enumerate(grad)], axis=0)


# In[ ]:


def maxima_divergence(masks_pats):
    with Pool(CPU) as pool:
        divs = pool.map(
            functools.partial(divergence, 
                              patient=pat), 
            sigmas
        )
        divs = -1 * asarray(divs) * mask 
        divs = divs.max(axis=0)
        divs_list.append(divs.copy())
    return divs_list


# 

# In[ ]:


def is_in(colour, labe, dng_colours):
    if colour in dng_colours:
        return labe == colour


def get_pure_isol(patient, mask, enh_dot):
    isolated = (patient > -600) * (mask > 0) * (enh_dot < 6) 
    labe, iso_nodules_num = label(isolated)
    volumes = bincount(labe.flatten())
    colours = where((volumes > ISOLATED_MIN_VOLUME) 
                & (volumes < ISOLATED_MAX_VOLUME))[0]
    
    isolated = zeros(isolated.shape).astype(bool)
    for colour in colours:
        isolated |= labe == colour
        
    return isolated, iso_nodules_num


def get_pure_j_va(patient, mask, enh_line, iso):
    juxtavascular = (patient > -600) * (mask > 0) * (enh_line > 150)
    j_va_candidates = (1 - juxtavascular) * (1 - iso)
    labe, j_va_nodules_num = label(j_va_candidates)

    volumes = bincount(labe.flatten())
    colours = where((volumes > JUXTAVASCULAR_MIN_VOLUME) 
                    & (volumes < ISOLATED_MAX_VOLUME))[0]
    j_va = zeros(juxtavascular.shape).astype(bool)
    for colour in colours:
        j_va |= labe == colour
    
    return j_va, j_va_nodules_num


def get_pure_j_pl(patient, mask, enh_dot):
    fixed_mask = morphology.binary_erosion(mask > 0,iterations=4)
    border_mask = fixed_mask * (1 - morphology.binary_erosion(fixed_mask > 0,iterations=4))
    juxtapleural = (patient > -400) * (border_mask > 0) * (enh_dot > 4)

    labe, j_pl_num = label(juxtapleural)
    volumes = bincount(labe.flatten())
    colours = where((volumes > JUXTAPLEURAL_MIN_VALUME) 
                    & (volumes < ISOLATED_MAX_VOLUME))[0]
    j_pl = zeros(juxtapleural.shape).astype(bool)
    for colour in colours:
        j_pl |= labe == colour
    return j_pl, j_pl_num


# In[ ]:


def get_pure_nodules(patient, mask, enh):
    """
    Here: 
    1 is for isolated
    2 is for j_va
    4 is for j_pl
    """
    iso, iso_num = get_pure_isol(patient, mask, enh[0])
    j_va, j_va_num = get_pure_j_va(patient, mask, enh[1], iso)
    j_pl, j_pl_num = get_pure_j_pl(patient, mask, enh[0])
    return 2 * j_va + iso + 4 * j_pl, (iso_num, j_va_num, j_pl_num)


# 

# In[ ]:





# 
