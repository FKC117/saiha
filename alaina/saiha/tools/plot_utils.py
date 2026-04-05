# d:/quantly/quanta/quantalytics/utils/plot_utils.py

import matplotlib
import matplotlib.pyplot as plt
import base64
import io
from contextlib import contextmanager

class PlotUtils:
    """A utility class for creating and handling matplotlib plots."""

    @staticmethod
    @contextmanager
    def setup_plotting():
        """
        A context manager to set up the matplotlib backend for non-interactive plotting
        and to ensure plots are closed properly.
        """
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Use non-interactive backend
        try:
            yield
        finally:
            plt.close('all')  # Close all figures to free memory
            matplotlib.use(original_backend)  # Restore original backend

    @staticmethod
    def fig_to_base64(fig: plt.Figure) -> dict:
        """
        Converts a matplotlib figure to a base64 encoded string
        AND attempts to extract structured data for frontend rendering (ApexCharts).
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # --- Confidence-Gated Extraction ---
        structured_data = PlotUtils.extract_structured_data(fig)
        
        return {
            "base64": img_str,
            "structured_data": structured_data,
            "confidence": "high" if structured_data else "low"
        }

    @staticmethod
    def extract_structured_data(fig: plt.Figure) -> dict:
        """
        Extracts raw data and metadata from a matplotlib figure for ApexCharts.
        Strict confidence rules: fall back to None if the plot is too complex.
        """
        try:
            # FORCE a render to populate tick labels and coordinates in headless backends
            fig.canvas.draw()
            
            axes = fig.get_axes()
            if not axes or len(axes) > 1: return None # No multi-axes for now
            
            ax = axes[0]
            
            # Metadata Extraction
            title = ax.get_title()
            x_label = ax.get_xlabel()
            y_label = ax.get_ylabel()
            
            # 1. Bar Chart Extraction (from Patches)
            if len(ax.patches) > 0 and len(ax.patches) < 100:
                # Extract labels from the axis ticks
                labels = [tick.get_text() for tick in ax.get_xticklabels()]
                # If labels are empty, try getting them from the patches themselves or the legend
                if not any(labels):
                    labels = [p.get_label() for p in ax.patches if not p.get_label().startswith('_')]
                
                values = [p.get_height() for p in ax.patches]
                
                # Cleanup: remove empty strings/nulls from labels if they correspond to non-bars
                # (Sometimes matplotlib has extra ticks)
                if len(labels) > len(values):
                    labels = labels[:len(values)]
                
                # If we have labels matching the bars, it's high confidence
                if any(labels) and len(labels) == len(values):
                    return {
                        "type": "bar",
                        "title": title,
                        "x_label": x_label,
                        "y_label": y_label,
                        "labels": labels,
                        "series": [{"name": y_label or "Value", "data": [float(v) for v in values]}]
                    }
                    
            # 2. Line Chart Extraction (from Lines)
            if len(ax.lines) == 1:
                line = ax.lines[0]
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                
                # Convert numpy arrays to lists
                import numpy as np
                if isinstance(x_data, np.ndarray): x_data = x_data.tolist()
                if isinstance(y_data, np.ndarray): y_data = y_data.tolist()
                
                # Convert to floats for JSON serialization
                y_data = [float(v) for v in y_data]
                
                return {
                    "type": "line",
                    "title": title,
                    "x_label": x_label,
                    "y_label": y_label,
                    "series": [{"name": y_label or "Trend", "data": y_data}],
                    "labels": [str(x) for x in x_data]
                }
                
        except Exception as e:
            # print(f"Extraction error: {e}")
            pass # Fail silently and rely on the high-quality image fallback
            
        return None