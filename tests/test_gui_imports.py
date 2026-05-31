"""Quick import verification for the new GUI modules."""
import sys
print(f"Python: {sys.version}")

try:
    print("Testing widget imports...")
    from gigaseal.gui.widgets.pandas_model import PandasModel
    print("  PandasModel OK")
    from gigaseal.gui.widgets.param_form import ParamFormWidget
    print("  ParamFormWidget OK")
    from gigaseal.gui.widgets.sweep_selector import SweepSelector
    print("  SweepSelector OK")

    print("Testing panel imports...")
    from gigaseal.gui.panels.file_panel import FilePanel
    print("  FilePanel OK")
    from gigaseal.gui.panels.analysis_panel import AnalysisPanel
    print("  AnalysisPanel OK")
    from gigaseal.gui.panels.results_panel import ResultsPanel
    print("  ResultsPanel OK")
    from gigaseal.gui.panels.plot_panel import PlotPanel
    print("  PlotPanel OK")

    print("Testing controller import...")
    from gigaseal.gui.controllers.analysis_controller import AnalysisController
    print("  AnalysisController OK")

    print("Testing app import...")
    from gigaseal.gui.app import MainWindow, main
    print("  MainWindow OK")

    print("\nAll imports successful!")
except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
