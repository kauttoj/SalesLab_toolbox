# Gglobal variables and hard-coded parameters
# By changing these, you can control the behavior of the program
def init():
    global INPUT_FILE
    global process_file
    global myprint
    global updateTextFun
    global SCREEN_SIZE
    global GSR_MINIMUM_RATE
    global USE_SUBFOLDER
    global VERSION
    global BANDPASS_FILTER
    global DATA_TO_EXTRACT

    # program version
    VERSION = '3.10.2019'

    # GUI related parameters and function handles
    INPUT_FILE=None
    process_file= None
    myprint = None
    updateTextFun = None

    # parameters related to data-analysis
    SCREEN_SIZE = {'w':1980,'h':1080} # presentation screen size in pixels
    GSR_MINIMUM_RATE = 10 # smallest allowed sampling rate [Hz] for GRS signal deconvolution (at least 5Hz required)
    USE_SUBFOLDER = True # put results into subfolders
    BANDPASS_FILTER = {'low':0.001,'high':5.0,'order':2} # lower and upper thresholds in Hz, order of filter as integer 1-5

    # Define data of interest. These are expected to be present in the raw exported textfile.
    # Note: Column names must match those by iMotions, may change with future updates
    DATA_TO_EXTRACT = {'GSR':{'SOURCE':'shimmer','COLUMNS':['Siemens']}, # only one for GSR, passed for deconvolution
                        'HEART':{'SOURCE':'shimmer','COLUMNS':['Beats/min']},
                        'ANNOTATION': {'SOURCE':'','COLUMNS':['PostMarker','Annotation']}, # annotation is special: HAs no specific events, column can be one of the two (not both!)
                        'EYE':{'SOURCE':'et','COLUMNS':['FixationX','FixationY','FixationSeq']},
                        'FACE':{'SOURCE':'affraw','COLUMNS':[
        'Number of faces', # this is an integer as 0,1,2...
        'Valence', # between -100 and 100
        'Smile', # this and all remaining should be already between 0-100
        'Attention',
        'Engagement',
        'Smirk',
        'Anger',
        'Sadness',
        'Disgust',
        'Joy',
        'Surprise',
        'Fear',
        'Contempt']}}