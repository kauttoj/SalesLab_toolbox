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
    global iMOTIONS_VERSION
    global BANDPASS_FILTER
    global DATA_TO_EXTRACT
    global SEPARATOR
    global GSR_DISCONTINUITY_LIMIT
    global GSR_MINIMUM_SEGMENT_LIMIT

    # program version
    VERSION = '19.1.2022'
    iMOTIONS_VERSION = 9

    # GUI related parameters and function handles
    INPUT_FILE=None
    process_file= None
    myprint = None
    updateTextFun = None
    linestyle = '.'

    # parameters related to data-analysis
    SCREEN_SIZE = {'w':1980,'h':1080} # presentation screen size in pixels
    GSR_MINIMUM_RATE = 10 # smallest allowed sampling rate [Hz] for GRS signal deconvolution (at least 5Hz required)
    GSR_DISCONTINUITY_LIMIT = 3  # [s] if larger, consider as different segment
    GSR_MINIMUM_SEGMENT_LIMIT = 20 # [s] if smaller, skip analysis
    USE_SUBFOLDER = True # put results into subfolders
    BANDPASS_FILTER = {'low':0.01,'high':5.0,'order':2} # lower and upper thresholds in Hz, order of filter as integer 1-5
    SEPARATOR = ';'

    # Define data of interest. These are expected to be present in the raw exported textfile.
    # Note: Column names must match those by iMotions, may change with future updates
    DATA_TO_EXTRACT = dict()
    DATA_TO_EXTRACT[7] = {'GSR':{'SOURCE':'shimmer','COLUMNS':['Siemens']}, # only one for GSR, passed for deconvolution
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

    '''
    Logic of iMotions version 8: 
    
    Data is no longer mixed
    
        #METADATA
        #Device  <--- [source label]
        #Category <--- Event Source or Data
        #Description
        #Unit
        #DataType
        #Data format
        #
        #DATA
        Row  <--- EventSource or [name]
    
    note: Sources are lower-cased!
    '''
    DATA_TO_EXTRACT[8] = {'GSR':{'SOURCE':'(shimmer)','CATEGORY':'','COLUMNS':['GSR Conductance CAL']}, # only one for GSR, passed for deconvolution
                        'HEART':{'SOURCE':'(shimmer)','CATEGORY':'','COLUMNS':['Heart Rate PPG ALG']},
                        #'ANNOTATION': {'SOURCE':'','COLUMNS':['PostMarker','Annotation']}, # annotation is special: HAs no specific events, column can be one of the two (not both!)
                        #'EYE':{'SOURCE':'','COLUMNS':['FixationX','FixationY','FixationSeq']},
                        'FACE':{'SOURCE':'affectiva affdex','CATEGORY':'','COLUMNS':[
        #'Number of faces', # this is an integer as 0,1,2...
        'Valence', # between -100 and 100
        'Smile', # this and all remaining should be already between 0-100
        'Attention',
        'Engagement',
        'Anger',
        'Sadness',
        'Disgust',
        'Joy',
        'Surprise',
        'Fear',
        'Contempt','Number of faces']}}

    DATA_TO_EXTRACT[9] = {'GSR':{'SOURCE':'Shimmer','CATEGORY':'','COLUMNS':['GSR Conductance CAL']}, # only one for GSR, passed for deconvolution
                        'HEART':{'SOURCE':'Shimmer','CATEGORY':'','COLUMNS':['Heart Rate PPG ALG']},
                        'STIMULUS':{'SOURCE':'SlideEvents','CATEGORY':'Slideshow','COLUMNS':['SourceStimuliName','SlideEvent']},
                        'ANNOTATION': {'SOURCE':'','CATEGORY':'','COLUMNS':['PostMarker','Annotation']}, # annotation is special: HAs no specific events, column can be one of the two (not both!)
                        'EYE':{'SOURCE':'R Analysis GazeAnalysis I-VT filter','CATEGORY':'Fixations','COLUMNS':['Fixation X','Fixation Y','Fixation Index']},
                        'FACE':{'SOURCE':'Affectiva AFFDEX','CATEGORY':'','COLUMNS':[
        #'Number of faces', # this is an integer as 0,1,2...
        'Valence', # between -100 and 100
        'Smile', # this and all remaining should be already between 0-100
        'Attention',
        'Engagement',
        'Anger',
        'Sadness',
        'Disgust',
        'Joy',
        'Surprise',
        'Fear',
        'Contempt','Number of faces']}}