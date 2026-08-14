"""Generate adaptive mesh for Rongelap data

Source for data in csv files: https://rdrr.io/cran/geoR/man/rongelap.html
Peter J. Diggle , Paulo J. Ribeiro: "Model based geostatistics", Springer 2007 https://link.springer.com/book/10.1007/978-0-387-48536-2

"""

import datetime
import numpy as np
import pandas as pd


def read_coastline(filename, h_min=0):
    """Read coastline from csv file

    Coarsens the coastline by dropping all points which have a
    distance of less than h_min

    Parameters
    ==========
    filename :
        name of file to read, points should be of the form x,y
    h_min :
        minimum distance between points

    Returns
    =======
        array of shape (n,2) with points on coastline
    """
    coastline_df = pd.read_csv(filename)
    coastline = np.stack([coastline_df[d].to_numpy() for d in ("x", "y")]).T
    p0 = coastline[0, :]
    coastline_coarse = [p0]
    for p in coastline:
        if np.linalg.norm(p - p0) > h_min:
            coastline_coarse.append(p)
            p0 = p
    return np.array(coastline_coarse)


def domain_bbox(coastline, padding=100):
    """Compute boundary of domain

    Parameters
    ==========
    coastline :
        array of shape (n,2) with coastline points
    padding :
        padding to at in all directions

    Returns
    =======
    Bounding box in the the form (x_min, x_max, y_min, y_max)
    """
    return (
        float(np.min(coastline[:, 0]) - padding),
        float(np.max(coastline[:, 0]) + padding),
        float(np.min(coastline[:, 1]) - padding),
        float(np.max(coastline[:, 1]) + padding),
    )


def generate_geometry(
    coastline,
    filename,
    padding=1000,
    h_min=50,
    h_max=400,
    d_transition=1000,
    d_halo=200,
):
    """Create geometry file for adaptively refined mesh

    The exact expression for the grid spacing is

           { h_min if x in I
    h(x) = { h_min if |x-x_c| < d_halo
           { (h_max-h_min)/d_transition if d_halo < |x-x_c| < d_halo + d_transition
           { h_max if |x-x_c| > d_halo + d_transition

    where x_c is the point on the coastline closest to x.

    Parameters
    ==========
    coastline :
        coastline points, array of shape (n,2)
    filename :
        name of geometry file to save to
    padding :
        space allocated around the island
    h_min :
        minimum grid spacing inside the island and within distance d_halo
    h_max :
        maximum grid spacing, to be used inside further away from coast
    d_transition :
        distance over which the grid spacing decays from h_min to h_max
    d_halo :
        halo around the island with grid spacing h_min
    """
    x_min, x_max, y_min, y_max = domain_bbox(coastline, padding=padding)
    with open(filename, "w", encoding="utf8") as f:
        now = datetime.datetime.now(tz=datetime.UTC)
        print("// File generated at ", str(now), file=f)
        # Domain boundary
        for j, (x, y) in enumerate(
            ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
        ):
            print(f"Point({j + 1}) = {{{x}, {y}, 0}};", file=f)
        for j in range(4):
            print(f"Line({j + 1:d}) = {{{j + 1:d},{(j + 1) % 4 + 1:d}}};", file=f)
        print("Curve Loop(1) = {1,2,3,4};", file=f)
        print("Plane Surface(1) = {1};", file=f)
        # Coastline
        offset_idx = 5
        for j, (x, y) in enumerate(coastline):
            print(f"Point({j + offset_idx}) = {{{x}, {y}, 0}};", file=f)
        n_points, _ = coastline.shape
        for j in range(n_points):
            print(
                f"Line({offset_idx + j:d}) = {{{offset_idx + j:d}, {offset_idx + ((j + 1) % n_points)}}};",
                file=f,
            )
        s = ",".join(str(j) for j in range(offset_idx, offset_idx + n_points))
        print(f"Curve Loop(2) = {{{s}}};", file=f)
        print(f"Curve {{{s}}} In Surface{{1}};", file=f)

        # Control grid spacing
        print(f"h_min = {h_min};", file=f)
        print(f"h_max = {h_max};", file=f)
        print(f"d_halo = {d_halo};", file=f)
        print(f"d_transition = {d_transition};", file=f)
        print("Field[1] = Distance;", file=f)
        print(f"Field[1].CurvesList = {{{s}}};", file=f)
        print("Field[1].Sampling = 100;", file=f)
        print("Field[2] = Threshold;", file=f)
        print("Field[2].InField = 1;", file=f)
        print(f"Field[2].LcMin = {h_min};", file=f)
        print(f"Field[2].LcMax = {h_max};", file=f)
        print(f"Field[2].DistMin = {d_halo};", file=f)
        print(f"Field[2].DistMax = {d_halo + d_transition};", file=f)
        print("Background Field = 2;", file=f)
        for j, boundary in enumerate(["bottom", "right", "top", "left"]):
            print(f'Physical Curve("{boundary}") = {{{j + 1:d}}};', file=f)
        print('Physical Surface("domain") = {1};', file=f)


def generate_vtk(coastline, filename):
    """Save coastline as a vtk file

    Parameters
    ==========
    coastline :
        coastline points, array of shape (n,2)
    filename :
        name of geometry file to save to
    """
    n_points, _ = coastline.shape
    with open(filename, "w", encoding="utf8") as f:
        print("# vtk DataFile Version 3.0", file=f)
        print("Rongelap coastline", file=f)
        print("ASCII", file=f)
        print("DATASET POLYDATA", file=f)
        print(f"POINTS {n_points} float", file=f)
        for x, y in coastline:
            print(f"{x:16.6f} {y:16.6f} 0", file=f)
        print(f"LINES 1 {n_points + 1:d}", file=f)
        s = " ".join(str(x) for x in range(n_points))
        print(f"{n_points} {s}", file=f)


if __name__ == "__main__":
    # File with csv data of coastline
    coastline_filename = "rongelap_coastline.csv"
    # Name of output file
    filename = "rongelap"
    # Padding to apply around coastline to obtain domain
    padding = 2000
    # Minimum grid spacing, to be used inside and
    # close to the island
    h_min = 50
    # Maximum grid spacing, to be used inside further away from coast
    h_max = 400
    # distance over which the grid spacing decays from h_min to h_max
    d_transition = 1000
    # halo around the island with grid spacing h_min
    d_halo = 200

    coastline = read_coastline(coastline_filename, h_min)

    generate_geometry(
        coastline,
        filename + ".geo",
        padding=padding,
        h_min=h_min,
        h_max=h_max,
        d_transition=d_transition,
        d_halo=d_halo,
    )
    print("Generated mesh geometry in {filename}.geo. Now run")
    print()
    print(f"   gmsh -2 {filename}.geo -o {filename}.msh")
    print()

    coastline_highres = read_coastline(coastline_filename)
    generate_vtk(coastline_highres, filename + ".vtk")
    print(f"Converted coastline to {filename}.vtk.")
    print()
