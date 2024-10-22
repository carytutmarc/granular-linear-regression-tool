# Constants
import os
import pandas as pd

# USER Set linear regression grid size and ranges
class GridObj:
    def __init__(self):
        self.START = 2   # Index of data point to start fit grid
        self.STARTMAX = None # Set the highest starting index to test
        self.INCREMENT = 10   # Increment i.e. grid size of fit steps
        self.LENGTHMIN = 4 * self.INCREMENT # Miniumn length of linear fit
        self.LENGTHMAX = None # Maximum allowed fit length
        self.ENDMAX = None    # Optional 0.0-1.0 value to lower max fit length

# USER Set main figure parameters
class FigObj:
    def __init__(self):
        self.title = "Best Fit Linear Regression Analysis" # Main composite figure title
 
# USER Set HEATMAP Variables and Parameters
class HeatmapConfig:
    def __init__(self):
        self.x_label = "Start Index"
        self.y_label = "End Index"
        self.title = "Granular Inverse Standard Error" # Heatmap title
        self.cmap = "plasma"
        self.cbar = True
        
# USER Set PLOT Variables and Parameters
class PlotConfig:
    # Configure dataset to analyze
    def load_dataframe(self):
        # Load desired dataset into pandas dataframe. Requires X (list), Y (list), and Sample (string) data
        df = pd.read_
        
        # Configure dataset parameters
        self.sample = 'Title' # Column where sample names are found
        self.xVar = 'Time_Minutes' # Column where X data is found
        self.yVar = 'Cover_Areas_MicronMicron' # Column where Y data is found
        self.x_label = 'Time [min]'
        self.y_label = r'Cell Cover Area [$\mu m^2$]'
        self.title = 'Wound Healing Assay' # Linear fit graph title
        self.slopeunits = 'µm²/min'
        
        # Normalization factor e.g. stage height in micrometers for cell wound healing divided by 60 mins to get per hour
        self.normalization = (df['Stage_Height_Micron'] / 60) * 2 # Normalization to be applied to slope
        self.ratelabel = 'Normalized Rate' #Annotation label for the rate value e.g. Proliferation Rate
        self.normalunits = 'µm/hour' #Normalized units applying normalization_factor in HeatmapConfig
        
        return df
    
    # Load a test data set of 2D cell wound healing data (Time and Cover Area) and calculate cell front velocity 
    def load_test_dataframe(self):
        # Read test data
        import ast
        df = pd.read_csv(os.path.join(os.getcwd(), 'tests', 'testdata.csv'))
        df['Time'] = df['Time'].apply(ast.literal_eval)
        df['Area'] = df['Area'].apply(ast.literal_eval)
        
        # Parameters for test data
        self.sample = 'Sample'   
        self.xVar = 'Time'
        self.yVar = 'Area'
        self.x_label = 'Time [min]'
        self.y_label = r'Cell Cover Area [$\mu m^2$]'
        self.title = 'Wound Healing Assay'
        self.slopeunits = 'µm²/min'
        
        # Normalization factor e.g. stage height in micrometers for cell wound healing divided by 60 mins to get per hour
        normalization_df = pd.DataFrame(index=df.index)
        normalization_df['Normalization'] = (850 / 60) * 2
        
        self.normalization = normalization_df['Normalization'] # Normalization to be applied to slope
        self.ratelabel = 'Normalized Rate' #Annotation label for the rate value e.g. Proliferation Rate
        self.normalunits = 'µm/hour' #Normalized units applying normalization_factor in HeatmapConfig
        
        return df