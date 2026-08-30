"""Interactive WebGL diamond geometry for the Streamlit predictor."""

from __future__ import annotations

from math import cos, pi, sin

import numpy as np
import plotly.graph_objects as go


_COLOUR_TINTS = {
    "D": "#fbffff",
    "E": "#f3fdff",
    "F": "#e9faff",
    "G": "#ddf5ff",
    "H": "#f1efd5",
    "I": "#f4e3ae",
    "J": "#f1cf78",
}

_CUT_PROFILES = {
    "Fair": (0.34, 0.49),
    "Good": (0.305, 0.55),
    "Very Good": (0.275, 0.595),
    "Premium": (0.235, 0.655),
    "Ideal": (0.255, 0.625),
}

_CLARITY_OPACITY = {
    "I1": 0.98,
    "SI2": 0.94,
    "SI1": 0.90,
    "VS2": 0.85,
    "VS1": 0.81,
    "VVS2": 0.77,
    "VVS1": 0.74,
    "IF": 0.70,
}

_INCLUSION_COUNTS = {
    "I1": 14,
    "SI2": 9,
    "SI1": 7,
    "VS2": 4,
    "VS1": 3,
    "VVS2": 2,
    "VVS1": 1,
    "IF": 0,
}


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _mix(colour: str, accent: str, amount: float, alpha: float) -> str:
    base_rgb = np.asarray(_hex_to_rgb(colour), dtype=float)
    accent_rgb = np.asarray(_hex_to_rgb(accent), dtype=float)
    mixed = np.rint(base_rgb * (1 - amount) + accent_rgb * amount).astype(int)
    return f"rgba({mixed[0]},{mixed[1]},{mixed[2]},{alpha:.2f})"


def _diamond_mesh(
    colour_grade: str,
    cut: str,
    x_mm: float,
    y_mm: float,
    z_mm: float,
    depth_percentage: float,
    table_percentage: float,
) -> tuple[
    np.ndarray,
    list[tuple[int, int, int]],
    list[str],
    dict[str, int],
    float,
    float,
]:
    """Create a proportion-aware round-brilliant-style solid in millimetres."""
    segments = 16
    crown_base, pavilion_break = _CUT_PROFILES.get(cut, _CUT_PROFILES["Ideal"])
    depth_adjustment = float(np.clip((depth_percentage - 61.8) * 0.0022, -0.055, 0.055))
    crown_fraction = float(np.clip(crown_base + depth_adjustment * 0.35, 0.18, 0.36))
    girdle_fraction = 0.035

    crown_height = z_mm * crown_fraction
    girdle_half = z_mm * girdle_fraction / 2
    pavilion_depth = max(z_mm - crown_height, z_mm * 0.52)
    top_z = crown_height
    culet_z = -pavilion_depth

    table_ratio = float(np.clip(table_percentage / 100, 0.36, 0.82))
    star_ratio = table_ratio + (1 - table_ratio) * 0.48
    pavilion_break_ratio = float(
        np.clip(pavilion_break + (depth_percentage - 61.8) * 0.003, 0.46, 0.74)
    )

    rings = [
        ("table", table_ratio, top_z, 0.0),
        ("star", star_ratio, crown_height * 0.53, pi / segments),
        ("girdle_top", 1.00, girdle_half, 0.0),
        ("girdle_bottom", 1.00, -girdle_half, 0.0),
        ("pavilion_break", pavilion_break_ratio, -pavilion_depth * 0.43, pi / segments),
        ("pavilion_main", 0.27, -pavilion_depth * 0.77, 0.0),
    ]

    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, top_z)]
    ring_starts: dict[str, int] = {}
    for name, radius_ratio, height, offset in rings:
        ring_starts[name] = len(vertices)
        for index in range(segments):
            angle = (2 * pi * index / segments) + offset
            vertices.append(
                (
                    (x_mm / 2) * radius_ratio * cos(angle),
                    (y_mm / 2) * radius_ratio * sin(angle),
                    height,
                )
            )
    culet_index = len(vertices)
    vertices.append((0.0, 0.0, culet_z))

    faces: list[tuple[int, int, int]] = []
    face_groups: list[str] = []
    table_start = ring_starts["table"]
    for index in range(segments):
        faces.append((0, table_start + index, table_start + (index + 1) % segments))
        face_groups.append("table")

    ring_pairs = [
        ("table", "star", "crown_upper"),
        ("star", "girdle_top", "crown_lower"),
        ("girdle_top", "girdle_bottom", "girdle"),
        ("girdle_bottom", "pavilion_break", "pavilion_upper"),
        ("pavilion_break", "pavilion_main", "pavilion_lower"),
    ]
    for upper_name, lower_name, group in ring_pairs:
        upper = ring_starts[upper_name]
        lower = ring_starts[lower_name]
        for index in range(segments):
            next_index = (index + 1) % segments
            if index % 2:
                faces.extend(
                    [
                        (upper + index, lower + index, upper + next_index),
                        (upper + next_index, lower + index, lower + next_index),
                    ]
                )
            else:
                faces.extend(
                    [
                        (upper + index, lower + index, lower + next_index),
                        (upper + index, lower + next_index, upper + next_index),
                    ]
                )
            face_groups.extend([group, group])

    pavilion_start = ring_starts["pavilion_main"]
    for index in range(segments):
        faces.append(
            (pavilion_start + index, culet_index, pavilion_start + (index + 1) % segments)
        )
        face_groups.append("pavilion_main")

    tint = _COLOUR_TINTS.get(colour_grade, _COLOUR_TINTS["G"])
    accents = ["#ffffff", "#8df6ff", "#5fc8ff", "#bbb8ff", "#effdff"]
    group_mix = {
        "table": 0.07,
        "crown_upper": 0.15,
        "crown_lower": 0.22,
        "girdle": 0.18,
        "pavilion_upper": 0.21,
        "pavilion_lower": 0.28,
        "pavilion_main": 0.20,
    }
    face_opacity = 0.88
    face_colours = []
    for index, group in enumerate(face_groups):
        light_variation = (index * 7) % len(accents)
        alpha = face_opacity - (0.12 if group == "girdle" else 0.02 * (index % 3))
        face_colours.append(
            _mix(
                tint,
                accents[light_variation],
                min(0.48, group_mix[group] + (index % 4) * 0.042),
                max(0.58, alpha),
            )
        )
    return (
        np.asarray(vertices),
        faces,
        face_colours,
        ring_starts,
        top_z,
        culet_z,
    )


def _structural_edge_trace(
    vertices: np.ndarray,
    ring_starts: dict[str, int],
) -> go.Scatter3d:
    """Draw the intentional facet structure without every triangulation diagonal."""
    segments = 16
    x_coords: list[float | None] = []
    y_coords: list[float | None] = []
    z_coords: list[float | None] = []

    def add_edge(start: int, end: int) -> None:
        x_coords.extend([vertices[start, 0], vertices[end, 0], None])
        y_coords.extend([vertices[start, 1], vertices[end, 1], None])
        z_coords.extend([vertices[start, 2], vertices[end, 2], None])

    for ring_start in ring_starts.values():
        for index in range(segments):
            add_edge(ring_start + index, ring_start + (index + 1) % segments)

    ring_pairs = [
        ("table", "star"),
        ("star", "girdle_top"),
        ("girdle_bottom", "pavilion_break"),
        ("pavilion_break", "pavilion_main"),
    ]
    for first_name, second_name in ring_pairs:
        first = ring_starts[first_name]
        second = ring_starts[second_name]
        for index in range(0, segments, 2):
            add_edge(first + index, second + index)

    table_start = ring_starts["table"]
    pavilion_start = ring_starts["pavilion_main"]
    culet_index = len(vertices) - 1
    for index in range(0, segments, 2):
        add_edge(0, table_start + index)
        add_edge(pavilion_start + index, culet_index)

    return go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode="lines",
        line={"color": "rgba(228, 249, 255, 0.22)", "width": 1.25},
        hoverinfo="skip",
        showlegend=False,
    )


def _luminous_stage_traces(
    x_mm: float,
    y_mm: float,
    girdle_z: float,
    brilliance: float,
) -> list[go.Scatter3d]:
    """Create restrained cyan light rings attached to the diamond girdle."""
    angles = np.linspace(0, 2 * pi, 181)
    traces: list[go.Scatter3d] = []
    for scale, width, alpha in [(1.015, 7.5, 0.08), (1.01, 3.2, 0.24), (1.004, 1.25, 0.70)]:
        traces.append(
            go.Scatter3d(
                x=(x_mm / 2) * scale * np.cos(angles),
                y=(y_mm / 2) * scale * np.sin(angles),
                z=np.full_like(angles, girdle_z),
                mode="lines",
                line={
                    "color": f"rgba(93, 230, 255, {alpha * brilliance:.3f})",
                    "width": width,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _sparkle_traces(
    vertices: np.ndarray,
    ring_starts: dict[str, int],
    x_mm: float,
    y_mm: float,
    z_mm: float,
    brilliance: float,
) -> list[go.Scatter3d]:
    """Add a few precise white-blue glints without turning the scene into glitter."""
    selected = [
        ring_starts["table"] + 2,
        ring_starts["star"] + 7,
        ring_starts["girdle_top"] + 12,
    ]
    points = vertices[selected]
    halo = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker={
            "size": [10 + 6 * brilliance, 8 + 4 * brilliance, 9 + 5 * brilliance],
            "color": [
                f"rgba(126,239,255,{.05 + .09 * brilliance:.3f})",
                f"rgba(255,255,255,{.04 + .07 * brilliance:.3f})",
                f"rgba(147,197,253,{.05 + .08 * brilliance:.3f})",
            ],
            "symbol": "diamond",
        },
        hoverinfo="skip",
        showlegend=False,
    )
    core = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker={
            "size": [2.8 + 1.3 * brilliance, 2.2 + .9 * brilliance, 2.5 + 1.0 * brilliance],
            "color": ["#ffffff", "#d9fbff", "#b6e9ff"],
            "symbol": "diamond",
        },
        hoverinfo="skip",
        showlegend=False,
    )

    ray_x: list[float | None] = []
    ray_y: list[float | None] = []
    ray_z: list[float | None] = []
    ray_lengths = [0.075, 0.055, 0.065]
    for point, length in zip(points, ray_lengths):
        dx = max(x_mm, y_mm) * length
        dz = z_mm * length * 1.45
        ray_x.extend([point[0] - dx, point[0] + dx, None, point[0], point[0], None])
        ray_y.extend([point[1], point[1], None, point[1], point[1], None])
        ray_z.extend([point[2], point[2], None, point[2] - dz, point[2] + dz, None])
    rays = go.Scatter3d(
        x=ray_x,
        y=ray_y,
        z=ray_z,
        mode="lines",
        line={
            "color": f"rgba(224, 252, 255, {.32 + .52 * brilliance:.3f})",
            "width": 1.4 + brilliance,
        },
        hoverinfo="skip",
        showlegend=False,
    )
    return [halo, rays, core]


def _measurement_traces(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    top_z: float,
    culet_z: float,
) -> list[go.Scatter3d]:
    offset = max(0.34, max(x_mm, y_mm) * 0.085)
    x_offset = x_mm / 2 + offset
    y_offset = -(y_mm / 2 + offset)
    label_z = top_z * 0.05

    line_x = [
        -x_mm / 2,
        x_mm / 2,
        None,
        -x_offset,
        -x_offset,
        None,
        x_offset,
        x_offset,
    ]
    line_y = [
        y_offset,
        y_offset,
        None,
        -y_mm / 2,
        y_mm / 2,
        None,
        0,
        0,
    ]
    line_z = [
        label_z,
        label_z,
        None,
        label_z,
        label_z,
        None,
        culet_z,
        top_z,
    ]

    line_trace = go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        line={"color": "rgba(111, 224, 255, 0.78)", "width": 2.5},
        hoverinfo="skip",
        showlegend=False,
    )
    label_trace = go.Scatter3d(
        x=[0, -x_offset, x_offset],
        y=[y_offset, 0, 0],
        z=[label_z, label_z, (top_z + culet_z) / 2],
        mode="text",
        text=[f"x  {x_mm:.2f} mm", f"y  {y_mm:.2f} mm", f"z  {z_mm:.2f} mm"],
        textfont={"color": "#bdefff", "size": 12, "family": "Arial"},
        hoverinfo="skip",
        showlegend=False,
    )
    return [line_trace, label_trace]


def _inclusion_trace(
    clarity: str,
    x_mm: float,
    y_mm: float,
    top_z: float,
    culet_z: float,
) -> go.Scatter3d | None:
    count = _INCLUSION_COUNTS.get(clarity, 0)
    if count == 0:
        return None
    rng = np.random.default_rng(2026 + count)
    radius = np.sqrt(rng.uniform(0.02, 0.50, count))
    angle = rng.uniform(0, 2 * pi, count)
    x_values = radius * (x_mm / 2) * np.cos(angle)
    y_values = radius * (y_mm / 2) * np.sin(angle)
    z_values = rng.uniform(culet_z * 0.45, top_z * 0.72, count)
    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="markers",
        marker={
            "size": {
                "I1": 5.0,
                "SI2": 4.2,
                "SI1": 3.6,
                "VS2": 3.0,
                "VS1": 2.6,
                "VVS2": 2.2,
                "VVS1": 1.9,
            }.get(clarity, 1.8),
            "color": "rgba(22, 38, 52, 0.72)",
        },
        hoverinfo="skip",
        showlegend=False,
    )


def create_diamond_figure(
    carat: float = 0.70,
    colour_grade: str = "G",
    cut: str = "Ideal",
    clarity: str = "VS2",
    x_mm: float = 5.70,
    y_mm: float = 5.71,
    z_mm: float = 3.53,
    depth_percentage: float = 61.8,
    table_percentage: float = 57.0,
    show_measurements: bool = True,
    brilliance: float = 0.82,
) -> go.Figure:
    """Return a live, proportion-aware and camera-animated WebGL figure."""
    (
        vertices,
        faces,
        face_colours,
        ring_starts,
        top_z,
        culet_z,
    ) = _diamond_mesh(
        colour_grade,
        cut,
        x_mm,
        y_mm,
        z_mm,
        depth_percentage,
        table_percentage,
    )
    triangle_indices = np.asarray(faces, dtype=int)
    clarity_opacity = _CLARITY_OPACITY.get(clarity, 0.86)
    brilliance = float(np.clip(brilliance, 0.15, 1.0))

    glow_vertices = vertices * np.asarray([1.022, 1.022, 1.018])
    glow_mesh = go.Mesh3d(
        x=glow_vertices[:, 0],
        y=glow_vertices[:, 1],
        z=glow_vertices[:, 2],
        i=triangle_indices[:, 0],
        j=triangle_indices[:, 1],
        k=triangle_indices[:, 2],
        color="#65e8ff",
        flatshading=False,
        opacity=0.025 + 0.085 * brilliance,
        lighting={
            "ambient": 1.0,
            "diffuse": 0.12,
            "specular": 0.0,
            "roughness": 1.0,
            "fresnel": 5.0,
        },
        hoverinfo="skip",
        showlegend=False,
        name="Diamond glow",
    )

    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangle_indices[:, 0],
        j=triangle_indices[:, 1],
        k=triangle_indices[:, 2],
        facecolor=face_colours,
        flatshading=True,
        opacity=clarity_opacity,
        lighting={
            "ambient": 0.22,
            "diffuse": 0.82,
            "specular": 1.40 + 0.58 * brilliance,
            "roughness": 0.09 - 0.065 * brilliance,
            "fresnel": 3.9 + 0.9 * brilliance,
        },
        lightposition={"x": 6.8, "y": 4.4, "z": 9.6},
        hovertemplate=(
            f"<b>{carat:.2f} ct · {cut}</b><br>"
            f"Colour {colour_grade} · Clarity {clarity}<br>"
            f"{x_mm:.2f} × {y_mm:.2f} × {z_mm:.2f} mm<br>"
            f"Depth {depth_percentage:.1f}% · Table {table_percentage:.1f}%"
            "<extra></extra>"
        ),
        name="Diamond",
    )

    traces: list[go.BaseTraceType] = [glow_mesh]
    traces.extend(_luminous_stage_traces(x_mm, y_mm, 0.0, brilliance))
    traces.extend([mesh, _structural_edge_trace(vertices, ring_starts)])
    traces.extend(
        _sparkle_traces(vertices, ring_starts, x_mm, y_mm, z_mm, brilliance)
    )
    inclusion_trace = _inclusion_trace(clarity, x_mm, y_mm, top_z, culet_z)
    if inclusion_trace is not None:
        traces.append(inclusion_trace)
    if show_measurements:
        traces.extend(_measurement_traces(x_mm, y_mm, z_mm, top_z, culet_z))

    centre_angle = pi / 4
    scan_width = pi * 0.30
    forward_scan = np.linspace(centre_angle - scan_width, centre_angle + scan_width, 42)
    reverse_scan = np.linspace(centre_angle + scan_width, centre_angle - scan_width, 42)[1:-1]
    scan_angles = np.concatenate([forward_scan, reverse_scan])
    frames = [
        go.Frame(
            name=f"orbit-{index}",
            layout=go.Layout(
                scene_camera={
                    "eye": {"x": 2.36 * cos(angle), "y": 2.36 * sin(angle), "z": 1.08},
                    "up": {"x": 0, "y": 0, "z": 1},
                }
            ),
        )
        for index, angle in enumerate(scan_angles)
    ]

    scene_span = max(9.2, x_mm * 1.34, y_mm * 1.34, z_mm * 1.72)
    half_span = scene_span / 2
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        height=440,
        margin={"l": 0, "r": 0, "t": 0, "b": 46},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dcecff"},
        showlegend=False,
        scene={
            "bgcolor": "rgba(0,0,0,0)",
            "xaxis": {
                "visible": False,
                "showbackground": False,
                "range": [-half_span, half_span],
            },
            "yaxis": {
                "visible": False,
                "showbackground": False,
                "range": [-half_span, half_span],
            },
            "zaxis": {
                "visible": False,
                "showbackground": False,
                "range": [-half_span, half_span],
            },
            "aspectmode": "cube",
            "dragmode": "turntable",
            "camera": {
                "eye": {"x": 1.68, "y": 1.68, "z": 1.02},
                "up": {"x": 0, "y": 0, "z": 1},
            },
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.5,
                "xanchor": "center",
                "y": -0.01,
                "yanchor": "bottom",
                "bgcolor": "rgba(10, 31, 49, 0.88)",
                "bordercolor": "rgba(103, 232, 249, 0.40)",
                "font": {"color": "#dff8ff", "size": 12},
                "buttons": [
                    {
                        "label": "◀▶  Left-right scan",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "mode": "immediate",
                                "frame": {"duration": 60, "redraw": True},
                                "transition": {"duration": 35, "easing": "linear"},
                            },
                        ],
                    },
                    {
                        "label": "Ⅱ  Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        uirevision=(
            f"diamond-{colour_grade}-{cut}-{clarity}-{x_mm:.2f}-{y_mm:.2f}-"
            f"{z_mm:.2f}-{depth_percentage:.1f}-{table_percentage:.1f}"
        ),
    )
    return fig
