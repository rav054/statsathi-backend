"""Publication charts rendered with Matplotlib / Seaborn."""

import io
import base64
from typing import Optional

import pandas as pd
import numpy as np
import scipy.stats as stats
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form
from fastapi.responses import StreamingResponse
import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt
import seaborn as sns

from ..auth import get_current_user
from ..figure_style import (PALETTES, PRESETS, STYLE, FONT_STACK, pretty_label, category_label_fit,
                            rotate_category_labels, EXPORT_MEDIA_TYPES, export_figure, figure_size,
                            figure_style, get_preset, normalize_format, preview_dpi, colormap_name, COLORS)
from . import chart_gallery
from . import slide_export
from ..models import User

from .common import read_uploaded_table

router = APIRouter()

def safe_float(val):
    if val is None or val == "" or str(val).strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def safe_bool(val, default=False):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if str(val).lower() in ("true", "1", "yes"):
        return True
    if str(val).lower() in ("false", "0", "no"):
        return False
    return default


def safe_int(val):
    if val is None or val == "" or str(val).strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _render_plot(
    file: UploadFile,
    plot_type: str,
    x_var: str,
    y_var: Optional[str] = None,
    hue_var: Optional[str] = None,
    bins: Optional[int] = None,
    kde: Optional[bool] = None,
    fit_reg: Optional[bool] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    palette: Optional[str] = None,
    legend_loc: Optional[str] = "best",
    show_grid: Optional[bool] = True,
    aspect_ratio: Optional[str] = "standard",
    text_color: Optional[str] = COLORS["text"],
    errorbar_toggle: Optional[bool] = False,
    errorbar_type: Optional[str] = "sd",
    title_font_size: Optional[str] = None,
    title_font_family: Optional[str] = None,
    label_font_size: Optional[str] = None,
    label_font_family: Optional[str] = None,
    tick_font_size: Optional[str] = None,
    tick_font_family: Optional[str] = None,
    xlim_min: Optional[str] = None,
    xlim_max: Optional[str] = None,
    ylim_min: Optional[str] = None,
    ylim_max: Optional[str] = None,
    x_interval: Optional[str] = None,
    y_interval: Optional[str] = None,
    x_vars_str: Optional[str] = None,
    y_vars_str: Optional[str] = None,
    download_format: str = "png",
    dpi: int = 150,
    preset: Optional[str] = None
):
    df = read_uploaded_table(file)

    # Validate variables
    for var in [x_var]:
        if var not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{var}' not found in dataset."
            )

    if y_var and y_var not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{y_var}' not found in dataset."
        )

    if hue_var and hue_var not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{hue_var}' not found in dataset."
        )

    cols_to_use = []
    pca_cols = []
    y_cols = []

    if y_vars_str:
        y_cols = list(dict.fromkeys([c.strip() for c in y_vars_str.split(",") if c.strip()]))

    if plot_type == "multiline":
        if not y_cols:
            raise HTTPException(status_code=400, detail="Multi-line plot requires at least one Y variable column.")
        cols_to_use = [x_var]
        for col in y_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in dataset.")
            cols_to_use.append(col)
    elif plot_type == "pcabiplot":
        if y_cols:
            pca_cols = y_cols
        else:
            pca_cols = list(df.select_dtypes(include=[np.number]).columns)
            if hue_var in pca_cols:
                pca_cols.remove(hue_var)
        if len(pca_cols) < 2:
            raise HTTPException(status_code=400, detail="PCA Biplot requires at least 2 numeric variables.")
        cols_to_use = []
        for col in pca_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in dataset.")
            cols_to_use.append(col)
    else:
        cols_to_use = [x_var]
        if y_cols:
            for col in y_cols:
                if col not in df.columns:
                    raise HTTPException(status_code=400, detail=f"Column '{col}' not found in dataset.")
                cols_to_use.append(col)
        elif y_var:
            cols_to_use.append(y_var)
    
    if hue_var and hue_var not in cols_to_use:
        cols_to_use.append(hue_var)

    # Globally deduplicate cols_to_use to avoid duplicate columns in df_clean
    cols_to_use_unique = []
    for col in cols_to_use:
        if col not in cols_to_use_unique:
            cols_to_use_unique.append(col)
    cols_to_use = cols_to_use_unique

    df_clean = df[cols_to_use].dropna()

    if len(df_clean) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset contains no valid observations after removing missing values for selected variables."
        )

    # Set up styling palette
    palette_map = {
        "sunset": ("#F97316", "Oranges"),
        "indigo": ("#4F46E5", "Purples"),
        "teal": ("#0D9488", "GnBu"),
        "crimson": ("#DC2626", "Reds"),
        "charcoal": ("#475569", "Greys"),
        "coolwarm": ("#4F46E5", "coolwarm"),
        "viridis": ("#4F46E5", "viridis"),
        "emerald": ("#10B981", "Greens"),
        "amber": ("#F59E0B", "YlOrBr"),
        "rose": ("#F43F5E", "RdPu"),
        "skyblue": ("#0EA5E9", "Blues"),
        "forest": ("#15803D", "YlGn"),
        "navy": ("#1E3A8A", "crest"),
        "spring": ("#E11D48", "spring"),
        "summer": ("#F59E0B", "summer"),
        "autumn": ("#D97706", "autumn"),
        "winter": ("#0284C7", "winter"),
    }
    default_colors = PALETTES[STYLE["default_palette"]]["colors"]
    if palette in PALETTES:
        # Publication palettes shared with the interactive charts (figure_style.json)
        sns_palette = PALETTES[palette]["colors"]
        primary_color, accent_color = sns_palette[0], sns_palette[1]
    elif palette in palette_map:
        # The older single-hue choices: one colour plus its own ramp for grouped charts
        primary_color, sns_palette = palette_map[palette]
        accent_color = "#F97316" if palette == "indigo" else default_colors[1]
    else:
        # No choice made: groups get the default palette's colours, not a ramp of one hue
        sns_palette = list(default_colors)
        primary_color, accent_color = default_colors[0], default_colors[1]

    # Width comes from the size preset (screen, journal column, poster); the ratio sets the height.
    aspect_map = {"wide": 0.5625, "square": 1.0, "standard": 0.75}
    figsize = figure_size(preset, aspect_map.get(aspect_ratio))
    # Font sizes chosen on screen are scaled so text keeps its proportion at the preset's width
    font_scale = get_preset(preset)["font_pt"] / PRESETS["screen"]["font_pt"]

    if plot_type in chart_gallery.CHARTS:
        # Modern chart types (raincloud, ridgeline, joint, mosaic, ...) build their own figure layout
        if palette in PALETTES:
            chart_colors = list(sns_palette)
        else:
            chart_colors = [primary_color, accent_color] + sns.color_palette(sns_palette, 6).as_hex()
        spec = chart_gallery.ChartSpec(
            x=x_var, y=y_var, hue=hue_var, y_vars=y_cols, colors=chart_colors, cmap=colormap_name(palette),
            figsize=figsize, font_scale=font_scale, fit_reg=safe_bool(fit_reg, False),
            show_grid=safe_bool(show_grid, True), legend_loc=legend_loc or "best",
            title=title, xlabel=xlabel, ylabel=ylabel, text_color=text_color or COLORS["text"],
            title_size=(safe_float(title_font_size) or 12) * font_scale,
            label_size=(safe_float(label_font_size) or 10) * font_scale,
            tick_size=(safe_float(tick_font_size) or 9) * font_scale,
            title_family=title_font_family or "sans-serif",
            label_family=label_font_family or "sans-serif",
            tick_family=tick_font_family or "sans-serif",
        )
        fig = None
        try:
            fig = chart_gallery.render(plot_type, df, spec)
            content, _ = export_figure(fig, download_format, dpi)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate custom chart: {str(e)}"
            )
        finally:
            if fig is not None:
                plt.close(fig)
        return io.BytesIO(content)

    known_types = {"boxplot", "histogram", "qqplot", "scatter", "pie", "line", "barplot", "violin",
                   "multiline", "pcabiplot"}
    if plot_type not in known_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown chart type '{plot_type}'. Choose one of: {', '.join(sorted(known_types))}."
        )

    plot_buf = io.BytesIO()

    try:
        fig, ax = plt.subplots(figsize=figsize)

        errorbar_toggle_bool = safe_bool(errorbar_toggle, False)
        show_grid_bool = safe_bool(show_grid, True)

        if plot_type == "boxplot":
            if y_var:
                sns.boxplot(
                    data=df_clean,
                    x=x_var,
                    y=y_var,
                    hue=hue_var,
                    # Colour means a group, never a rank: without a grouping variable every box is the same colour
                    palette=sns_palette if hue_var else None,
                    color=primary_color if not hue_var else None,
                    ax=ax,
                    width=0.45,
                    linewidth=0.9,
                    fliersize=3,
                    legend=True if hue_var else False
                )
                # The points are the data behind the box, so they stay one ink colour even when grouped
                ink = {level: COLORS["text"] for level in df_clean[hue_var].dropna().unique()} if hue_var else None
                sns.stripplot(
                    data=df_clean,
                    x=x_var,
                    y=y_var,
                    hue=hue_var,
                    palette=ink,
                    color=None if hue_var else COLORS["text"],
                    size=3.5,
                    jitter=0.1,
                    alpha=0.45,
                    dodge=bool(hue_var),
                    ax=ax,
                    legend=False
                )
            else:
                try:
                    df_clean[x_var] = pd.to_numeric(df_clean[x_var])
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Variable '{x_var}' must be numeric to draw a boxplot without groups."
                    )
                sns.boxplot(y=df_clean[x_var], color=primary_color, width=0.3, ax=ax)
                sns.stripplot(y=df_clean[x_var], color=COLORS["text"], size=3.5, jitter=0.05, alpha=0.45, ax=ax)

        elif plot_type == "histogram":
            try:
                df_clean[x_var] = pd.to_numeric(df_clean[x_var])
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Variable '{x_var}' must be numeric to draw a histogram."
                )
            
            sns.histplot(
                data=df_clean, 
                x=x_var, 
                hue=hue_var, 
                bins=bins if bins else "auto", 
                kde=safe_bool(kde, True), 
                palette=sns_palette if hue_var else None,
                color=primary_color if not hue_var else None,
                alpha=0.6, 
                ax=ax
            )

        elif plot_type == "qqplot":
            try:
                df_clean[x_var] = pd.to_numeric(df_clean[x_var])
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Variable '{x_var}' must be numeric to draw a Q-Q plot."
                )
            stats.probplot(df_clean[x_var], dist="norm", plot=ax)
            lines = ax.get_lines()
            if len(lines) >= 2:
                lines[0].set_color(primary_color)
                lines[0].set_markersize(4)
                lines[0].set_alpha(0.7)
                lines[1].set_color(accent_color)
                lines[1].set_linewidth(2)

        elif plot_type == "scatter":
            if not y_var:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Scatter plot requires both X and Y variables."
                )
            try:
                df_clean[x_var] = pd.to_numeric(df_clean[x_var])
                df_clean[y_var] = pd.to_numeric(df_clean[y_var])
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Variables '{x_var}' and '{y_var}' must be numeric for a scatter plot."
                )

            sns.scatterplot(
                data=df_clean, 
                x=x_var, 
                y=y_var, 
                hue=hue_var,
                palette=sns_palette if hue_var else None,
                color=primary_color if not hue_var else None,
                alpha=0.75,
                edgecolor=COLORS["background"],
                linewidth=0.4,
                ax=ax
            )

            if safe_bool(fit_reg, False):
                sns.regplot(
                    data=df_clean, 
                    x=x_var, 
                    y=y_var, 
                    scatter=False,
                    color=primary_color if not hue_var else accent_color,
                    line_kws={"linewidth": 1.8},
                    ax=ax,
                    label="Trend"
                )

        elif plot_type == "pie":
            counts = df_clean[x_var].value_counts()
            if len(counts) > 8:
                top_counts = counts.iloc[:7]
                others_sum = counts.iloc[7:].sum()
                top_counts['Others'] = others_sum
                counts = top_counts
            
            colors = sns.color_palette(sns_palette, len(counts))
            ax.pie(
                counts, 
                labels=[str(label) for label in counts.index], 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=colors,
                textprops={'fontsize': (safe_float(tick_font_size) or 9) * font_scale, 'color': text_color},
                wedgeprops={'edgecolor': '#FFFFFF', 'linewidth': 1}
            )
            ax.axis('equal')

        elif plot_type == "line":
            if not y_var:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Line graph requires both X and Y variables."
                )
            df_sorted = df_clean.sort_values(by=x_var)
            
            err_bar = None
            if errorbar_toggle_bool:
                if errorbar_type == "sd":
                    err_bar = ("sd", 1)
                elif errorbar_type == "se":
                    err_bar = ("se", 1)
                elif errorbar_type == "ci":
                    err_bar = ("ci", 95)

            # Grouped lines also differ in dash pattern and marker, so they stay apart in print,
            # in greyscale and for a reader who cannot separate the hues
            sns.lineplot(
                data=df_sorted,
                x=x_var,
                y=y_var,
                hue=hue_var,
                style=hue_var,
                markers=True if hue_var else None,
                dashes=True if hue_var else False,
                marker=None if hue_var else "o",
                errorbar=err_bar,
                palette=sns_palette if hue_var else None,
                color=primary_color if not hue_var else None,
                ax=ax
            )

        elif plot_type == "barplot":
            if not y_var:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bar chart requires both X and Y variables."
                )
            
            err_bar = None
            if errorbar_toggle_bool:
                if errorbar_type == "sd":
                    err_bar = ("sd", 1)
                elif errorbar_type == "se":
                    err_bar = ("se", 1)
                elif errorbar_type == "ci":
                    err_bar = ("ci", 95)

            sns.barplot(
                data=df_clean,
                x=x_var,
                y=y_var,
                hue=hue_var,
                errorbar=err_bar,
                palette=sns_palette if hue_var else None,
                color=primary_color if not hue_var else None,
                capsize=0.12,
                err_kws={"linewidth": 1.0, "color": COLORS["text"]},
                edgecolor="none",
                ax=ax
            )

        elif plot_type == "violin":
            if not y_var:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Violin plot requires both X and Y variables."
                )
            sns.violinplot(
                data=df_clean,
                x=x_var,
                y=y_var,
                hue=hue_var,
                split=True if hue_var else False,
                inner="quartile",
                palette=sns_palette if hue_var else None,
                color=primary_color if not hue_var else None,
                ax=ax
            )

        elif plot_type == "multiline":
            df_sorted = df_clean.sort_values(by=x_var)
            colors = sns.color_palette(sns_palette, len(y_cols))
            for i, col in enumerate(y_cols):
                sns.lineplot(
                    data=df_sorted,
                    x=x_var,
                    y=col,
                    label=col,
                    color=colors[i],
                    marker='o' if len(df_sorted) < 50 else None,
                    ax=ax
                )

        elif plot_type == "pcabiplot":
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            
            df_pca = df_clean[pca_cols].dropna()
            if len(df_pca) < 3:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="PCA Biplot requires at least 3 valid observations."
                )
            
            X_scaled = StandardScaler().fit_transform(df_pca)
            pca = PCA(n_components=2)
            X_projected = pca.fit_transform(X_scaled)
            
            pc_df = pd.DataFrame(X_projected, columns=["PC1", "PC2"], index=df_pca.index)
            if hue_var and hue_var in df_clean.columns:
                pc_df[hue_var] = df_clean.loc[df_pca.index, hue_var]
                sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue=hue_var, palette=sns_palette, alpha=0.7, ax=ax)
            else:
                sns.scatterplot(data=pc_df, x="PC1", y="PC2", color=primary_color, alpha=0.7, ax=ax)
            
            loadings = pca.components_
            max_pc1 = np.max(np.abs(X_projected[:, 0]))
            max_pc2 = np.max(np.abs(X_projected[:, 1]))
            scale_factor = min(max_pc1, max_pc2) * 1.5 if min(max_pc1, max_pc2) > 0 else 1.0
            
            for i, col_name in enumerate(pca_cols):
                arrow_x = loadings[0, i] * scale_factor
                arrow_y = loadings[1, i] * scale_factor
                ax.arrow(
                    0, 0, arrow_x, arrow_y, 
                    color=text_color, alpha=0.7, 
                    head_width=0.04 * scale_factor, 
                    head_length=0.06 * scale_factor, 
                    length_includes_head=True
                )
                ax.text(
                    arrow_x * 1.15, arrow_y * 1.15, col_name, 
                    color=text_color, ha='center', va='center', 
                    fontweight='bold', fontsize=8 * font_scale
                )
            
            var_explained = pca.explained_variance_ratio_ * 100
            ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% Variance)")
            ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% Variance)")
            
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            ax.set_xlim(x_min * 1.25, x_max * 1.25)
            ax.set_ylim(y_min * 1.25, y_max * 1.25)

        # Apply fonts and titles
        title_font_val = (safe_float(title_font_size) or 12) * font_scale
        label_font_val = (safe_float(label_font_size) or 10) * font_scale
        tick_font_val = (safe_float(tick_font_size) or 9) * font_scale

        title_family = title_font_family if title_font_family else "sans-serif"
        label_family = label_font_family if label_font_family else "sans-serif"
        tick_family = tick_font_family if tick_font_family else "sans-serif"

        font_family_map = {
            "sans-serif": FONT_STACK + ["DejaVu Sans"],
            "serif": ["DejaVu Serif", "Times New Roman", "Georgia", "serif"],
            "monospace": ["Courier New", "DejaVu Sans Mono", "Consolas", "monospace"]
        }
        
        title_font_name = font_family_map.get(title_family, ["sans-serif"])
        label_font_name = font_family_map.get(label_family, ["sans-serif"])
        tick_font_name = font_family_map.get(tick_family, ["sans-serif"])

        if title:
            ax.set_title(title, fontsize=title_font_val, fontweight="bold", pad=15, color=text_color, family=title_font_name)
        else:
            default_titles = {
                "boxplot": f"Box Plot of {y_var if y_var else x_var}",
                "histogram": f"Distribution of {x_var}",
                "qqplot": f"Normal Q-Q Plot of {x_var}",
                "scatter": f"Scatter Plot: {y_var} vs {x_var}",
                "pie": f"Pie Chart of {x_var}",
                "line": f"Line Graph: {y_var} vs {x_var}",
                "barplot": f"Bar Chart: {y_var} vs {x_var}",
                "violin": f"Violin Plot: {y_var} vs {x_var}",
                "multiline": f"Multi-Line Plot of {x_var}",
                "pcabiplot": "PCA Biplot (PC1 vs PC2)"
            }
            ax.set_title(default_titles.get(plot_type, "Statistical Plot"), fontsize=title_font_val, fontweight="bold", pad=15, color=text_color, family=title_font_name)

        if plot_type != "pie":
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=label_font_val, color=text_color, family=label_font_name)
            elif plot_type == "pcabiplot":
                var_explained = pca.explained_variance_ratio_ * 100
                ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% Variance)", fontsize=label_font_val, color=text_color, family=label_font_name)
            else:
                ax.set_xlabel(pretty_label(x_var), fontsize=label_font_val, color=text_color, family=label_font_name)

            if plot_type == "pcabiplot":
                var_explained = pca.explained_variance_ratio_ * 100
                ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% Variance)", fontsize=label_font_val, color=text_color, family=label_font_name)
            elif y_var:
                if ylabel:
                    ax.set_ylabel(ylabel, fontsize=label_font_val, color=text_color, family=label_font_name)
                else:
                    ax.set_ylabel(pretty_label(y_var), fontsize=label_font_val, color=text_color, family=label_font_name)
            elif plot_type == "boxplot" and not y_var:
                ax.set_ylabel("Values", fontsize=label_font_val, color=text_color, family=label_font_name)
            elif plot_type == "qqplot":
                ax.set_ylabel("Sample Quantiles", fontsize=label_font_val, color=text_color, family=label_font_name)

            # Gridlines only across the measured axis; a category axis gets none
            if not show_grid_bool:
                ax.grid(False)
            else:
                category_x = plot_type in ("boxplot", "barplot", "violin") and bool(y_var)
                ax.grid(True, axis="y" if category_x or plot_type == "histogram" else "both")
                ax.set_axisbelow(True)

            for l in ax.get_xticklabels():
                l.set_fontsize(tick_font_val)
                l.set_family(tick_font_name)
                l.set_color(text_color)
            for l in ax.get_yticklabels():
                l.set_fontsize(tick_font_val)
                l.set_family(tick_font_name)
                l.set_color(text_color)

            ax.tick_params(axis='both', labelcolor=text_color)

            # Manual axis limits
            x_min_val = safe_float(xlim_min)
            x_max_val = safe_float(xlim_max)
            y_min_val = safe_float(ylim_min)
            y_max_val = safe_float(ylim_max)

            if x_min_val is not None or x_max_val is not None:
                ax.set_xlim(left=x_min_val, right=x_max_val)
            if y_min_val is not None or y_max_val is not None:
                ax.set_ylim(bottom=y_min_val, top=y_max_val)

            # Manual tick step size intervals
            import matplotlib.ticker as ticker
            x_int_val = safe_float(x_interval)
            y_int_val = safe_float(y_interval)

            if x_int_val is not None:
                try:
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_int_val))
                except Exception:
                    pass
            if y_int_val is not None:
                try:
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_int_val))
                except Exception:
                    pass

            # Rotate only category labels, and only when they really would overlap: the widths are
            # measured on the figure rather than guessed from the number of characters
            if x_var in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[x_var]):
                if category_label_fit(fig, ax) != "flat":
                    rotate_category_labels(ax)

            # Legend
            if legend_loc == "none":
                if ax.get_legend():
                    ax.get_legend().remove()
            else:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    leg = ax.legend(
                        loc=legend_loc,
                        prop={'family': tick_font_name, 'size': 8.5 * font_scale}
                    )
                    if leg:
                        for text in leg.get_texts():
                            text.set_color(text_color)
                        if leg.get_title():
                            leg.get_title().set_color(text_color)

        plt.tight_layout()
        content, _ = export_figure(fig, download_format, dpi)
        plot_buf = io.BytesIO(content)
        plt.close(fig)

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate custom chart: {str(e)}"
        )

    return plot_buf




def generate_plot_internal(preset: Optional[str] = None, **options):
    """Render a chart in the StatSathi figure style for the chosen size preset and palette."""
    palette = options.get("palette")
    with figure_style(preset, palette if palette in PALETTES else None):
        return _render_plot(preset=preset, **options)


@router.post("/plot")

def analyze_plot(
    file: UploadFile = File(...),
    plot_type: str = Form(...),
    x_var: str = Form(...),
    y_var: Optional[str] = Form(None),
    hue_var: Optional[str] = Form(None),
    bins: Optional[int] = Form(None),
    kde: Optional[bool] = Form(None),
    fit_reg: Optional[bool] = Form(None),
    title: Optional[str] = Form(None),
    xlabel: Optional[str] = Form(None),
    ylabel: Optional[str] = Form(None),
    palette: Optional[str] = Form(None),  # a key from figure_style.json; None uses the default palette
    legend_loc: Optional[str] = Form("best"),
    show_grid: Optional[bool] = Form(True),
    aspect_ratio: Optional[str] = Form("standard"),
    text_color: Optional[str] = Form(COLORS["text"]),
    errorbar_toggle: Optional[bool] = Form(False),
    errorbar_type: Optional[str] = Form("sd"),
    title_font_size: Optional[str] = Form(None),
    title_font_family: Optional[str] = Form(None),
    label_font_size: Optional[str] = Form(None),
    label_font_family: Optional[str] = Form(None),
    tick_font_size: Optional[str] = Form(None),
    tick_font_family: Optional[str] = Form(None),
    xlim_min: Optional[str] = Form(None),
    xlim_max: Optional[str] = Form(None),
    ylim_min: Optional[str] = Form(None),
    ylim_max: Optional[str] = Form(None),
    x_interval: Optional[str] = Form(None),
    y_interval: Optional[str] = Form(None),
    x_vars_str: Optional[str] = Form(None),
    y_vars_str: Optional[str] = Form(None),
    preset: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    plot_buf = generate_plot_internal(
        file=file,
        plot_type=plot_type,
        x_var=x_var,
        y_var=y_var,
        hue_var=hue_var,
        bins=bins,
        kde=kde,
        fit_reg=fit_reg,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        palette=palette,
        legend_loc=legend_loc,
        show_grid=show_grid,
        aspect_ratio=aspect_ratio,
        text_color=text_color,
        errorbar_toggle=errorbar_toggle,
        errorbar_type=errorbar_type,
        title_font_size=title_font_size,
        title_font_family=title_font_family,
        label_font_size=label_font_size,
        label_font_family=label_font_family,
        tick_font_size=tick_font_size,
        tick_font_family=tick_font_family,
        xlim_min=xlim_min,
        xlim_max=xlim_max,
        ylim_min=ylim_min,
        ylim_max=ylim_max,
        x_interval=x_interval,
        y_interval=y_interval,
        x_vars_str=x_vars_str,
        y_vars_str=y_vars_str,
        preset=preset,
        download_format="png",
        dpi=preview_dpi(preset)
    )
    img_b64 = base64.b64encode(plot_buf.getvalue()).decode('utf-8')
    return {"plot": img_b64}


def _download_as_slide(file, *, plot_type, x_var, y_var, hue_var, bins, title, palette,
                       y_vars_str, preset, **figure_options):
    """One PowerPoint slide for this chart, editable where PowerPoint has an equivalent chart type."""
    df = read_uploaded_table(file)
    if x_var not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Column '{x_var}' not found in dataset.")

    y_cols = [c.strip() for c in (y_vars_str or "").split(",") if c.strip()]
    data = slide_export.series_for(df, plot_type, x_var=x_var, y_var=y_var, hue_var=hue_var,
                                   bins=bins or 10, y_cols=y_cols)

    image = None
    if data is None:
        # No chart object for this type - place the publication figure and the numbers behind it
        if plot_type in slide_export.IMAGE_ONLY:
            data = slide_export.five_number_summary(df, x_var, y_var)
        # an unknown plot type is rejected by the renderer below, which owns that list
        image = generate_plot_internal(
            file=file, plot_type=plot_type, x_var=x_var, y_var=y_var, hue_var=hue_var, bins=bins,
            title=title, palette=palette, y_vars_str=y_vars_str, preset=preset,
            download_format="png", dpi=200, **figure_options
        ).getvalue()

    buffer = slide_export.build_deck(
        {"plot_type": plot_type, "title": title, "palette": palette}, data, image)
    filename = f"statsathi_plot_{plot_type}.pptx"
    return StreamingResponse(buffer, media_type=slide_export.MEDIA_TYPE,
                             headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.post("/plot/download")
def download_plot(
    file: UploadFile = File(...),
    plot_type: str = Form(...),
    x_var: str = Form(...),
    y_var: Optional[str] = Form(None),
    hue_var: Optional[str] = Form(None),
    bins: Optional[int] = Form(None),
    kde: Optional[bool] = Form(None),
    fit_reg: Optional[bool] = Form(None),
    title: Optional[str] = Form(None),
    xlabel: Optional[str] = Form(None),
    ylabel: Optional[str] = Form(None),
    palette: Optional[str] = Form(None),  # a key from figure_style.json; None uses the default palette
    legend_loc: Optional[str] = Form("best"),
    show_grid: Optional[bool] = Form(True),
    aspect_ratio: Optional[str] = Form("standard"),
    text_color: Optional[str] = Form(COLORS["text"]),
    errorbar_toggle: Optional[bool] = Form(False),
    errorbar_type: Optional[str] = Form("sd"),
    title_font_size: Optional[str] = Form(None),
    title_font_family: Optional[str] = Form(None),
    label_font_size: Optional[str] = Form(None),
    label_font_family: Optional[str] = Form(None),
    tick_font_size: Optional[str] = Form(None),
    tick_font_family: Optional[str] = Form(None),
    xlim_min: Optional[str] = Form(None),
    xlim_max: Optional[str] = Form(None),
    ylim_min: Optional[str] = Form(None),
    ylim_max: Optional[str] = Form(None),
    x_interval: Optional[str] = Form(None),
    y_interval: Optional[str] = Form(None),
    x_vars_str: Optional[str] = Form(None),
    y_vars_str: Optional[str] = Form(None),
    download_format: str = Form("png"),
    dpi: int = Form(300),
    preset: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    # PowerPoint is not a Matplotlib format: the slide carries a chart object built from the data
    # itself, so it stays editable in PowerPoint instead of being a picture of a result.
    if (download_format or "").strip().lower() in ("pptx", "ppt", "powerpoint"):
        return _download_as_slide(
            file, plot_type=plot_type, x_var=x_var, y_var=y_var, hue_var=hue_var, bins=bins,
            title=title, palette=palette, y_vars_str=y_vars_str, preset=preset,
            kde=kde, fit_reg=fit_reg, xlabel=xlabel, ylabel=ylabel, legend_loc=legend_loc,
            show_grid=show_grid, aspect_ratio=aspect_ratio, text_color=text_color,
            errorbar_toggle=errorbar_toggle, errorbar_type=errorbar_type,
            x_vars_str=x_vars_str,
        )

    try:
        download_format = normalize_format(download_format)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    dpi = max(72, min(int(dpi), 1200))
    plot_buf = generate_plot_internal(
        file=file,
        plot_type=plot_type,
        x_var=x_var,
        y_var=y_var,
        hue_var=hue_var,
        bins=bins,
        kde=kde,
        fit_reg=fit_reg,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        palette=palette,
        legend_loc=legend_loc,
        show_grid=show_grid,
        aspect_ratio=aspect_ratio,
        text_color=text_color,
        errorbar_toggle=errorbar_toggle,
        errorbar_type=errorbar_type,
        title_font_size=title_font_size,
        title_font_family=title_font_family,
        label_font_size=label_font_size,
        label_font_family=label_font_family,
        tick_font_size=tick_font_size,
        tick_font_family=tick_font_family,
        xlim_min=xlim_min,
        xlim_max=xlim_max,
        ylim_min=ylim_min,
        ylim_max=ylim_max,
        x_interval=x_interval,
        y_interval=y_interval,
        x_vars_str=x_vars_str,
        y_vars_str=y_vars_str,
        preset=preset,
        download_format=download_format,
        dpi=dpi
    )
    
    media_type = EXPORT_MEDIA_TYPES[download_format]
    extension = "jpg" if download_format == "jpeg" else download_format
    filename_out = f"statsathi_plot_{plot_type}.{extension}"
    headers = {
        "Content-Disposition": f"attachment; filename={filename_out}"
    }
    return StreamingResponse(plot_buf, media_type=media_type, headers=headers)
