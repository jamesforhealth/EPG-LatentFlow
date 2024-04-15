from pyqtgraph import AxisItem

class TimeAxisItem(AxisItem):
    def __init__(self, *args, **kwargs):
        self.sample_rate = kwargs.pop("sample_rate", 100)  # Default sample rate is 100 if not specified
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        # Override the method to generate custom tick labels
        return [f"{value/self.sample_rate:.1f}" for value in values]