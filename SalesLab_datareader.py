import pandas
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import copy
import os
from scipy import signal
import Saleslab_settings
import csv

# GLOBAL PARAMETERS #######################################
INPUT_FILE=None
PROFILE = False  # enable profiling, useful in debugging
###########################################################

pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

# parse string to allow quoting to work
def lineparser(linestr,delimiter):
    X = list(csv.reader([linestr],delimiter = delimiter,doublequote = False,skipinitialspace = True,quoting = csv.QUOTE_MINIMAL))
    return X[0]

# deprecated, now only use ver 9
def read_ver8_data(INPUT_FILE,delimiter,decimal):
    DEVICE_row = detect_header_rows(INPUT_FILE, separator=delimiter, identifier='#Device')
    CATEGORY_row = detect_header_rows(INPUT_FILE, separator=delimiter, identifier='#Category')
    DATA_row = detect_header_rows(INPUT_FILE, separator=delimiter, identifier='#DATA')+1
    def isvalid(arr):
        assert not(any(["|".find(x)>-1 for x in arr])),'Data contains illegal pipe-character!'
        return arr
    with open(INPUT_FILE,'r',encoding='utf-8') as f:
        line = f.readline()
        i=0
        while line:
            i+=1
            s = line.strip().split(delimiter)
            if i == DEVICE_row:
                DEVICES = isvalid(s[1:])
            if i == CATEGORY_row:
                CATEGORIES = isvalid(s[1:])
            if i == DATA_row:
                COLUMNS = isvalid(s[1:])
                break
            line = f.readline()
    new_names = ['|'.join(x) for x in list(zip(DEVICES,CATEGORIES,COLUMNS))]
    new_names = [x.replace('Affdex Facial Expression','Data') for x in new_names]
    new_names = [x.replace('Affdex Emotion','Data') for x in new_names]
    table = pandas.read_csv(INPUT_FILE, sep=delimiter, header=0, skiprows=DATA_row-1,low_memory=False,encoding='utf-8',index_col=False,names=new_names,usecols=range(1,len(new_names)+1),decimal=decimal)
    return table

# data reader, finds special lines and startpoint of actual data
def read_ver9_data(INPUT_FILE,delimiter,decimal):
    DEVICE_row = detect_header_rows(INPUT_FILE, separator=delimiter, identifier='#Device')
    CATEGORY_row = detect_header_rows(INPUT_FILE, separator=delimiter, identifier='#Category')
    DATA_row = detect_header_rows(INPUT_FILE, separator=delimiter, identifier='#DATA')+1
    def isvalid(arr):
        assert not(any(["|" in x for x in arr])),'Data contains illegal pipe-character!'
        return arr
    with open(INPUT_FILE,'r',encoding='utf-8') as f:
        line = f.readline()
        i=0
        while line:
            i+=1
            #s = line.strip().split(delimiter)
            s = lineparser(line,delimiter)
            if i == DEVICE_row:
                DEVICES = isvalid(s[1:])
            if i == CATEGORY_row:
                CATEGORIES = isvalid(s[1:])
            if i == DATA_row:
                COLUMNS = isvalid(s[1:])
                break
            line = f.readline()
    assert len(DEVICES)==len(CATEGORIES)==len(COLUMNS),"Different number of devices (%i), categories (%i) and columns (%i)!" % (len(DEVICES),len(CATEGORIES),len(COLUMNS))
    new_names = ['|'.join(x) for x in list(zip(DEVICES,CATEGORIES,COLUMNS))]
    #for old in ['Affdex Facial Expression','Affdex Emotion','GSR']:
    #    new_names = [x.replace(old,'Data') for x in new_names]
    table = pandas.read_csv(INPUT_FILE, sep=delimiter, header=0, skiprows=DATA_row-1,low_memory=False,encoding='utf-8',index_col=False,names=new_names,usecols=range(1,len(new_names)+1),decimal=decimal)
    assert all([x in table.columns for x in new_names]),"Not all new names in data!"

    return table

# helper function to get column names
def find_column_name(cols,targets,extra_target=None):
    ind = []
    for target in targets:
        for x in cols:
            if target in x:
                if extra_target!=None:
                    if extra_target in x:
                        ind.append(x)
                else:
                    ind.append(x)
    return ind

# helper function to get events
def get_events(event_set):
    ind = set()
    for x in event_set:
        y = x.split('|')
        for z in y:
            ind.add(z)
    return ind

# helper function to convert pandas frame to numpy array
def pandas_to_numpy(table):
    d={}
    for col in table.columns:
        try:
            vals = np.array(table[col],dtype=np.float32)
            if np.count_nonzero(~np.isnan(vals))>10:
                d[col] = vals
        except ValueError as error:
            vals = np.array(table[col],dtype=str)
            vals[table[col].isna()]=''
            if np.count_nonzero(vals!="")>0:
                d[col] = vals
    return d

# helper function to test monotonicity of array
def monotonic(x):
    dx = np.diff(x)
    dx = dx[~np.isnan(dx)]
    return np.all(dx <= 0) or np.all(dx >= 0)

# helper function to detect how many null header lines there are OR find specific lines
def detect_header_rows(file,separator='\t',identifier=''):
    with open(file,'r',encoding='utf-8') as f:
        line = f.readline()
        cnt = 1
        cols=-1
        while line:
            s = line.strip().split(separator)
            if s[0]==identifier:
                return cnt
            if len(s)>0 and len(s)<=cols and s[0][0]!='#':
                return cnt-2
            cols = len(s)
            line = f.readline()
            cnt += 1
    assert 0,"Did not find any valid data rows in file!"

# butterworth highpass filter
def butter_bandpass_filter(data,lowcut,highcut,fs,order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high],btype='band',analog=False,output='ba')
    y = signal.filtfilt(b, a, data)
    return y

# process GSR using ledapy signal decomposition
def process_gsr_signal(t,y):
    import ledapy

    valid_points = y>0 # data must be positive
    n1 = len(y)
    y = y[valid_points]
    t = t[valid_points]

    assert np.count_nonzero(y<=0)==0,"Non-positive GSR values remaining!"

    n2 = len(y)
    if n2<n1:
        print('...removed %i bad points from GSR data' % (n1-n2))

    median_dt = np.median(np.diff(t)) # original
    original_rate = 1.0 /median_dt

    discontinuity = list(np.where(np.diff(t)>=Saleslab_settings.GSR_DISCONTINUITY_LIMIT)[0])
    indices_to_process = [(0, len(t) - 1)]
    if len(discontinuity)>0:
        discontinuity = [-1]+discontinuity+[len(t)-1]
        indices_to_process = []
        for i in range(0,len(discontinuity)-1):
            if t[discontinuity[i+1]]-t[discontinuity[i]+1] >= Saleslab_settings.GSR_MINIMUM_SEGMENT_LIMIT:
                indices_to_process.append(((discontinuity[i]+1),(discontinuity[i+1])))
        print('!! Found %i discontinuity points, split data into %i parts (omitted %i too short)' % (len(discontinuity)-2,len(indices_to_process),len(discontinuity)+1-len(indices_to_process)) )

    tt1,yy1,phasic1,tonic1,yy_filtered1 = np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)

    for start_ind,end_ind in indices_to_process:
        # get equidistant signal
        tt_filtered = np.arange(t[start_ind], t[end_ind],median_dt)
        assert all(np.diff(t[start_ind:(end_ind+1)])<Saleslab_settings.GSR_DISCONTINUITY_LIMIT),"Segment failed discontinuity check (BUG)"
        yy = np.interp(tt_filtered,t[start_ind:(end_ind+1)],y[start_ind:(end_ind+1)])

        assert np.count_nonzero(yy < 0) == 0, "Negative GSR values found after interpolation!"

        yy_filtered = butter_bandpass_filter(yy-np.mean(yy),Saleslab_settings.BANDPASS_FILTER['low'],Saleslab_settings.BANDPASS_FILTER['high'],original_rate,order=Saleslab_settings.BANDPASS_FILTER['order'])

        #assert 5<sampling_rate<1000,"sampling rate of GSR was %i, does not make sense!" % sampling_rate
        N_points = 2950 # ledalab max points is 3000, try that first
        N_opt = 3 # number of optimization runs, between 0 and 4
        dt = (t[end_ind]-t[start_ind])/(N_points-1)
        new_rate = int(np.minimum(30,np.maximum(Saleslab_settings.GSR_MINIMUM_RATE,int(1.0/dt))))
        dt = 1.0 / new_rate
        tt = np.arange(t[start_ind], t[end_ind],dt)

        N_points = len(tt)

        assert np.abs(tt[-1]-t[end_ind])<2*dt,'Last timepoint difference too large! (BUG)'

        if 1:
            # simple numpy linear
            yy = np.interp(tt,t,y)
            yy_filtered = np.interp(tt,tt_filtered,yy_filtered)
        else:
            # fancy scipy nonlinear
            from scipy.interpolate import interp1d
            fun = interp1d(t,y,kind='cubic')
            yy = fun(tt)
            #print('Sampling rate %iHz and optimizations %i: ' % (new_rate, N_opt),end='')
        phasic,tonic = ledapy.runner.getResult(yy,'phasicdata+tonicdata',new_rate,downsample=1,optimisation=N_opt)
        print('...original GSR sampling rate %.1fHz (median) was interpolated into %.1fHz (%i datapoints, removed last %.4fs)' % (original_rate,new_rate,N_points,t[-1]-tt[-1]))
        assert phasic is not None, 'phasic signal decomposition failed!'

        tt1 = np.append(tt1,tt)
        yy1 = np.append(yy1,yy)
        phasic1 = np.append(phasic1,phasic)
        tonic1 = np.append(tonic1,tonic)
        yy_filtered1 = np.append(yy_filtered1,yy_filtered)

    return tt1,yy1,phasic1,tonic1,yy_filtered1

# process and plot EYE fixations
def plot_eye_data(t,fix_id,fix_x,fix_y,fname):
    from pygazeanalyser.gazeplotter import draw_fixations, draw_heatmap, draw_scanpath

    DISPSIZE = (int(Saleslab_settings.SCREEN_SIZE['w']),int(Saleslab_settings.SCREEN_SIZE['h']))

    # load image name, saccades, and fixations
    #imgname = data[trialnr][header.index("image")]
    #saccades = edfdata[trialnr]['events']['Esac']  # list of [starttime, endtime, duration,x,y]
    #fixations = edfdata[trialnr]['events']['Efix']  # list of [starttime, endtime, duration, endx, endy]

    # paths
    imagefile = None
    scatterfile = fname + "_fixations"
    heatmapfile =  fname + "_heatmap"
    #scanpathfile = INPUT_FILE_ROOT + "scanpath.png"

    # make fixation list
    fixations = []
    t=copy.deepcopy(t)*10000 # from seconds to 1/10000 of a second!
    for id in range(int(np.nanmin(fix_id)),int(np.nanmax(fix_id)+1)):
        ind = np.where(fix_id==id)[0]
        if len(ind)>0:
            fixations.append([t[ind[0]],t[ind[-1]],t[ind[-1]]-t[ind[0]],np.median(fix_x[ind]),np.median(fix_y[ind])])

    # plot fixations
    draw_fixations(fixations, DISPSIZE, imagefile=imagefile, durationsize=True, durationcolour=True, alpha=0.5,savefilename=scatterfile)
    # plot heatmap
    draw_heatmap(fixations, DISPSIZE, imagefile=imagefile, durationweight=True, alpha=0.5, savefilename=heatmapfile)
    # plot scanpath
    #fig2 = draw_scanpath(fixations, saccades, DISPSIZE, imagefile=imagefile, alpha=0.5, savefilename=scanpathfile)

# extract annotation events, results into a list of (starttime,endtime) blocks
def get_annotation_events(t,signals):
    best_anno = get_best_annotator(signals)
    event_dict = {}
    for key in signals.keys():
        event_dict[key] = {'type':best_anno==key,'events':{}}
        y=signals[key]
        all_events = list(np.unique(y))
        all_events = [x for x in all_events if x!='nan' and len(x)>2]
        if len(all_events)>0 and len(all_events)<100:
            for event in all_events:
                event_dict[key]['events'][event]=[]
                start = None
                last = None
                for i in range(len(t)):
                    if y[i] == event and i<len(t)-1:
                        if start is None:
                            start = i
                        last=i
                    else:
                        if start is not None and last is not None:
                            event_dict[key]['events'][event].append((t[start],t[last]))
                            start = None
                            last = None
    return event_dict

# add annotations as patches into given plot
def add_annotations(ax,annotation_events1,add_label=True):
    if len(annotation_events1)==0:
        return

    import matplotlib.patches as patches
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    cols = ('b','g','r','c','m','y','#E9AF8E','#C78E49','lime','tomato','olive','peru','gold','royalblue') #
    k=-1

    annotation_events=None
    for key in annotation_events1.keys():
        if annotation_events1[key]['type']==1:
            annotation_events=annotation_events1[key]['events']
            break
    # sort based on initial onset time
    mintimes = {}
    for event_name,events in annotation_events.items():
        mintimes[event_name]=np.inf
        for event in events:
            mintimes[event_name] = np.minimum(mintimes[event_name],event[0])
    mintimes = [x[0] for x in sorted(list(mintimes.items()),key=lambda x:x[1])]

    for event_name in mintimes:
        events = annotation_events[event_name]
        k+=1
        if k==len(cols):
            print('!!! More unique events (%i) than colors (%i), skipping all remaining events in plotting!!!' % (len(annotation_events),len(cols)))
            return
        for event in events:
            t_start = max(x_lims[0],event[0])
            t_end = min(x_lims[1],event[1])
            if t_start<t_end:
                p = patches.Rectangle((t_start,y_lims[0]),t_end-t_start,y_lims[1]-y_lims[0],color=cols[k],alpha = 0.20)
                ax.add_patch(p)
                if add_label:
                    t=plt.text((t_end + t_start) / 2,y_lims[0]+(y_lims[1]-y_lims[0])*0.04, event_name, ha='center',va='bottom', fontsize=11,rotation=90)
                    t.set_alpha(.60)

# write data as text file
def write_data(file,data,annotation_events,media_offset):
    t = data['time']
    annot_timeseries = {}
    for key in annotation_events.keys():
        t_anno = ['' for x in range(len(t))]
        #label_anno = set()
        for i in range(len(t)):
            for event_name,events in annotation_events[key]['events'].items():
                for event in events:
                    if (event[0] <= t[i] <= event[1]):
                        t_anno[i] = event_name
                        break
                    elif (i>0):
                        if (t[i-1]<= event[0] <= t[i]):
                            t_anno[i] = event_name
                            break
                        if (t[i-1]<= event[1] <= t[i]):
                            t_anno[i-1] = event_name
                            break

        annot_timeseries[key] = t_anno
    cols = [x for x in data['responses'].keys() if len(data['responses'][x])==len(t)] # only with same number of timepoints
    with open(file,'w',encoding="utf-8") as f:
        if media_offset == None:
            f.write("time[sec]\t")
        else:
            f.write("mediatime[sec]\t")
        for c in cols:
            f.write("%s\t" % c)
        for key in annot_timeseries.keys():
            f.write("%s\t" % key)
        f.write("\n")
        for i in range(len(t)):
            f.write("{}\t".format(t[i]))
            for c in cols:
                f.write("{}\t".format(data['responses'][c][i]))
            for key,x in annot_timeseries.items():
                f.write("{}\t".format(x[i]))
            f.write("\n")

# print progress information to console and GUI (if present)
def myprint(txt,end='\n',file=None):
    if hasGUI and Saleslab_settings.updateTextFun is not None:
        Saleslab_settings.updateTextFun(txt + end)
    print(txt,end=end)
    if file != None:
        file.write(txt + end)
        file.flush()

# test if all elements are NaN
def is_full_nan(d):
    n=0
    for key,val in d.items():
        if val.dtype.type is np.str_: # strings
            n+=np.sum(~(val == 'nan'))
        else: # numbers
            n+=np.sum(~np.isnan(val))
        if n>0:
            return False
    return True

# for some reason, shimmer events contain duplicate strings. Remove them.
def remove_duplicate(mystr):
    if len(mystr)>2 and len(mystr)%2 == 1:
        halfpoint = int(np.floor(len(mystr)/2.0))
        s1 = mystr[0:halfpoint]
        s2 = mystr[(halfpoint+1):]
        if s1==s2:
            if 'shimmer' not in s1.lower():
                print('!! Warning: Duplicate source for NON-shimmer source (weird) !!')
            return s1
    return mystr

# find starting point of the media and return offset compared to global time
def get_media_offset(X):
    X = X[np.logical_or(~np.isnan(X[:, 1]),~np.isnan(X[:, 0])), :]
    #if np.isnan(X[0,1]):  # only MediaTime is NaN,
    #    X = X[~np.isnan(X[:,1]),:] # remove all rows with NaN MediaTime, remaining is the offset
    #else: # assume zero useless, take the first nonzero
    X = X[X[:,1]>0,:]
    if X.shape[0]<10 or np.isnan(X[0,0]) or np.isnan(X[0,1]):
        myprint('Warning: Failed to detect MediaTime start reliably, data not synched to media!')
        return None
    return X[0, 0]-X[0,1] # this is the global timepoint where the media starts

# clean overlapping annotations, if present. Each moment can only contain one annotation label.
def clean_annotations(responses):
    r = list(responses.keys())[0]
    for i in range(len(responses[r])):
        ind = responses[r][i].find("|")
        if ind>-1:
            responses[r][i] = responses[r][i][0:ind]
    return responses

# try to detect iMotions version
def detect_version(infile):
    with open(infile,'r',encoding="utf-8") as f:
        for line in f:
            if 'iMotions version' in line:
                s = line.split(';')
                s = [x[0] for x in s if len(x)>0 and x[0].isnumeric()]
                if len(s)>0:
                    return int(s[0])
    return None

# try to detect column separator and decimal delimiter
def detect_delimiters(file_path, bytes = 4096):
    sniffer = csv.Sniffer()
    data = open(file_path, "r").read(bytes)
    delimiter = sniffer.sniff(data).delimiter
    separator = '.'
    if delimiter != ',':
        data = open(file_path, "r",encoding='utf-8').read()
        n_comma = data.count(',')
        n_dot = data.count('.')
        if n_dot<n_comma:
            separator=','
    return delimiter,separator

# find annotator that has most non-zero elements, we'll use that in plotting
def get_best_annotator(signals):
    best_n=-1
    best=None
    for key in signals.keys():
        try:
            n = np.sum(~np.isnan(signals[key]))
        except:
            n = np.sum(signals[key]!='')
        if n>best_n:
            best=key
            best_n=n
    return best

# function to process single file
def process_file(DATA_TO_EXTRACT_base):
    DATA_TO_EXTRACT = DATA_TO_EXTRACT_base[Saleslab_settings.iMOTIONS_VERSION]
    INPUT_FILE = Saleslab_settings.INPUT_FILE

    assert os.path.isfile(INPUT_FILE),"Input file \'%s\' not found!" % INPUT_FILE

    print('Input file was \'%s\', starting processing' % INPUT_FILE)

    # read dataset as Pandas frame
    assert any(INPUT_FILE[-4:]==item for item in ['.txt','.csv']),'Input file should be of type .TXT or .CSV!'

    INPUT_FILE_ROOT,file_name = os.path.split(INPUT_FILE)
    INPUT_FILE_ROOT += os.sep

    if INPUT_FILE_ROOT[0] == os.sep:
        INPUT_FILE_ROOT = INPUT_FILE_ROOT[1:]

    RESULT_FOLDER = INPUT_FILE_ROOT
    if Saleslab_settings.USE_SUBFOLDER: # collect results into new subfolders (recommended)
        RESULT_FOLDER = INPUT_FILE_ROOT + file_name[:-4] + "_datareader_output" + os.sep
        os.makedirs(RESULT_FOLDER,exist_ok=True)
    assert os.path.isdir(RESULT_FOLDER),"Output directory '%s' does not exist!" % RESULT_FOLDER

    # open a text file for reporting
    try:
        report_file = open(RESULT_FOLDER + file_name[:-4] + "_datareader_report.txt",'w',encoding="utf-8")
    except:
        myprint('Failed to open a text file for reporting!')
        report_file=None

    # try to detect version and delimiter
    version = detect_version(INPUT_FILE)
    if version is None or version<7:
        version = Saleslab_settings.iMOTIONS_VERSION
    separator,decimal = detect_delimiters(INPUT_FILE)
    if not(separator in [';',',','\t']):
        separator = Saleslab_settings.SEPARATOR

    # read datafile using Pandas
    skiprows=0
    try:
        myprint('\nReading file %s' % file_name,file=report_file)
        if version==7:
            skiprows = detect_header_rows(INPUT_FILE) # try to detect how many null rows before data
            table = pandas.read_csv(INPUT_FILE, sep=separator,header=0, skiprows=range(skiprows),low_memory=False,encoding='utf-8',index_col=False,decimal=decimal)#, usecols=COLUMNS_OF_INTEREST)
            if 'EventSource' not in table:
                myprint('No "EventSource" column found, not a valid dataset! Exiting...', file=report_file)
                assert False
        elif version==8:
            table = read_ver9_data(INPUT_FILE,separator,decimal)
        elif version==9:
            table = read_ver9_data(INPUT_FILE,separator,decimal)
        else:
            raise AssertionError("Unknown version!")
    except:
        myprint("!! FAILED to parse file, not expected RAW iMotions data format !!",file=report_file)
        raise "FAILED to read file %s into Pandas table!" % INPUT_FILE

    # should have some data, otherwise just stop here
    if table.shape[0]<100 or table.shape[1]<2:
        myprint("Less than 100 rows and/or columns, not a valid dataset! Exiting...",file=report_file)
        assert False

    if 'iMotions|Timestamp|Timestamp' in table.columns:
        timestamp_column = 'iMotions|Timestamp|Timestamp'
    elif '|Timestamp|Timestamp' in table.columns:
        timestamp_column = '|Timestamp|Timestamp'
    else:
        raise Exception('No timestamp column found, cannot parse data!')

    # get event sources
    COLUMNS = list(table.columns)

    media_offset = None
    if 'MediaTime' in table:
        X=np.array(table[['Timestamp','MediaTime']])
        media_offset = get_media_offset(X)

    # figure out which column contains annotations
    if 'PostMarker' in table and 'Annotation' not in table:
        DATA_TO_EXTRACT['ANNOTATION']['COLUMNS'] = ['PostMarker']
    elif 'PostMarker' not in table and 'Annotation' in table:
        DATA_TO_EXTRACT['ANNOTATION']['COLUMNS'] = ['Annotation']
    elif 'PostMarker' in table and 'Annotation' in table:
        assert False,"Data contains both PostMarker and Annotation, cannot process both"

    # get all valid data sources, could be multiple matching the same SOURCE pattern
    myprint('Parsing event sources',file=report_file)

    if version==7:
        EVENTS = np.array(table['EventSource']) # data event source
        all_EVENTS = get_events(set(EVENTS)) # separate events, could be multiple in same row!
        DATA_TO_EXTRACT_new = {}
        for key, value in DATA_TO_EXTRACT.items():
            if len(value['SOURCE'])>0: # has specified sources
                exact_sources = [event for i, event in enumerate(all_EVENTS) if value['SOURCE'] in event.lower()]
                if len(exact_sources)>0:
                    DATA_TO_EXTRACT_new[key]=[] # collect all into a list
                    for source in exact_sources:
                        source = remove_duplicate(source)
                        DATA_TO_EXTRACT_new[key].append({'SOURCE':source,'COLUMNS':DATA_TO_EXTRACT[key]['COLUMNS']})
            else: # no specified sources, pass as is
                DATA_TO_EXTRACT_new[key]=[{'SOURCE': '', 'COLUMNS': DATA_TO_EXTRACT[key]['COLUMNS']}]
    if version>7:
        all_EVENTS = table.columns
        DATA_TO_EXTRACT_new = {}
        for key, value in DATA_TO_EXTRACT.items():
            arr = [event for event in all_EVENTS if
                   (value['SOURCE'] in event.split('|')[0] and
                    value['CATEGORY'] in event.split('|')[1] and
                    any([x in event.split('|')[2] for x in value['COLUMNS']]))
                   ]
            exact_sources = list(np.unique(arr))
            unique_sources = list(np.unique([x.split('|')[0] for x in exact_sources]))
            DATA_TO_EXTRACT_new[key] = []
            for unique_source in unique_sources:
                d = []
                for exact_source in exact_sources:
                    if unique_source in exact_source:
                        d.append(exact_source)
                DATA_TO_EXTRACT_new[key].append(d)

    # this is the dict of those sources found in our table as recognized by pre-defined patterns
    DATA_TO_EXTRACT = copy.deepcopy(DATA_TO_EXTRACT_new) # create a new local copy

    myprint('Parsed OK! Data has %i rows (starting at %i), %i columns and %i unique sources' % (table.shape[0],skiprows+1,table.shape[1],len(DATA_TO_EXTRACT)),file=report_file)

    # doing some profiling?
    if PROFILE:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    # phase 1: collect all relevant data as numpy arrays
    # idea is that for each source type, we have single time-column and array of responses. Responses need to be in single table as they could be connected (e.g., X and Y coordinates of EYE).
    DATA = {}
    n_sources = 0
    myprint('Finding sources', file=report_file)
    for key, source_list in DATA_TO_EXTRACT.items():
        sub_sources = len(source_list)
        DATA[key] = []
        for value in source_list:
            cols = value if isinstance(value,list) else [value]
            event_rows = list(np.where((~table[cols].isna()).sum(axis=1)>0)[0]) # get indices with at least one value
            if len(event_rows) < 1:
                myprint('less than 10 events, skipping this source')
                continue
            cols.append(timestamp_column)
            if len(cols) > 1:  # must have at least two columns (always has TimeStamp)
                partial_table = table.iloc[event_rows]
                partial_table = partial_table[cols]
                # convert into dict of dense numpy arrays
                responses = pandas_to_numpy(partial_table.loc[:, partial_table.columns != timestamp_column])
                if key == "ANNOTATION":
                    if len(responses) > 1:
                        myprint(
                            "!! Multiple non-empty annotations columns found, cannot include annotations. Check your data !!")
                        continue
                    if len(responses) == 1:  # this is expected result
                        responses = clean_annotations(responses)  # remove overlapping annotations
                # test if all data are nans, omit if yes
                if not is_full_nan(responses):
                    d = {'time': np.array(partial_table[timestamp_column], dtype=np.float32), 'responses': responses}
                    DATA[key].append(d)
                    myprint('...found source %s (%i signals), raw data extracted (%i points)' % (
                    key,len(cols)-1,d['time'].shape[0]), file=report_file)
                    n_sources += 1

    assert n_sources>0,'!!! Could not find ANY valid data sources for this file, check your file !!!'

    DATA = {k:v for k,v in DATA.items() if len(v)>0}

    # get offset timepoint and set it to zero
    offset=None
    for i,key in enumerate(DATA.keys()):
        for key_set in range(len(DATA[key])):
            if offset == None:
                offset = DATA[key][key_set]['time'][0]
                endtime = np.nanmax(DATA[key][key_set]['time'])
            else:
                offset = min(offset,DATA[key][key_set]['time'][0])
                endtime = max(endtime,np.nanmax(DATA[key][key_set]['time']))

    # if we have MediaTime offset, use that instead to make media starttime the origin of time
    if media_offset is None or np.abs((media_offset-offset)/1000)>10:
        myprint('Using offset %.3fsec' % (offset/1000))
    else:
        offset = media_offset
        myprint('Using MediaTime offset (%.3fsec): Data is synched with media' % (media_offset/1000))

    myprint("Data timespan was %.2fmin, offset %.2fsec" % ((endtime-offset)/1000/60,offset/1000),file=report_file)

    # Phase 2: shift offset, remove all redundant timepoints via median
    myprint('Extracting events',file=report_file)
    for i,key in enumerate(DATA.keys()):
        for signal in DATA[key]:
            myprint('...extracting events for %s (%i signals)..' % (key,len(signal['responses'])),end='',file=report_file)
            signal['time'] -= offset
            signal['time'] = signal['time']/1000.0
            t = signal['time']
            inds = {}
            k=0
            val=0
            new_time = np.array([],dtype=np.float32)
            # extract all points with equal time
            while k<len(t):
                ref_time = t[k]
                kk=k+1
                while kk<len(t):
                    if t[kk]>ref_time:
                        break
                    kk+=1
                inds[val]=tuple(range(k,kk))
                new_time = np.append(new_time,t[k])
                k = kk
                val += 1
            signal['time'] = new_time
            assert monotonic(signal['time']),"Time array is not monotonic!"

            # get data for bins, take MEDIAN if more than one
            for response,y in signal['responses'].items():
                if np.isreal(y[0]):
                    # real type data
                    new_y = np.zeros(val,dtype=np.float32)
                    for i in range(len(inds)):
                        if len(inds[i])>1:
                            new_y[i] = np.nanmedian(y.take(inds[i])) # take median
                        else:
                            new_y[i] = y[inds[i]]
                else:
                    # other data, MEDIAN makes no sense
                    new_y = np.zeros(val,dtype=object)
                    for i in range(len(inds)):
                        new_y[i] = y[inds[i][0]] # we just take the first
                assert len(new_time)==len(new_y),"refined timeseries different length! (BUG)"
                signal['responses'][response]=new_y
            myprint(' %i samples' % len(signal['time']),file=report_file)

    # Phase 3: Process refined data and make preview plots

    # first make sure ANNOTATION is analyzed first, if present. It is added to plots.
    allkeys = list(DATA.keys())
    annotation_events = {}
    if 'STIMULUS' in allkeys:
        allkeys.pop(allkeys.index('STIMULUS'))
        allkeys = ['STIMULUS'] + allkeys
    elif 'ANNOTATION' in allkeys:
        allkeys.pop(allkeys.index('ANNOTATION'))
        allkeys = ['ANNOTATION'] + allkeys

    # iterate over data sources, add ANNOTATION if available
    for key in allkeys:
        for key_set,signals in enumerate(DATA[key]):
            myprint('Analyzing data for %s (#%i)' % (key,key_set+1),file=report_file)
            if key == 'ANNOTATION':
                #r = get_best_annotator(signals['responses'])  # response with most data
                t = signals['time']
                #y = signals['responses'][r]
                annotation_events = get_annotation_events(t,signals['responses'])
                signals['annotation_events'] = annotation_events
                myprint('...total %i annotations' % (len(annotation_events)),file=report_file)

            if key == 'STIMULUS':
                #r = get_best_annotator(signals['responses'])  # response with most data
                t = signals['time']
                #y = signals['responses'][r]
                annotation_events = get_annotation_events(t, signals['responses'])
                signals['annotation_events'] = annotation_events
                myprint('...total %i annotations' % (len(annotation_events)),file=report_file)

            if key == 'GSR':
                # compute phasic (fast) part of signal
                assert len(signals['responses']) == 1, "GSR should only contain single response (micro-Siemens), found %i" % len(
                    signals['responses'])
                r = list(signals['responses'].keys())[0]  # get whatever response name
                myprint('...computing phasic signal from raw GSR',file=report_file)

                #signals['responses'][r] = signals['responses'][r]
                new_time, new_y, phasic,tonic,hp_filtered = process_gsr_signal(signals['time'],signals['responses'][r])

                # set bad points into NaN
                phasic[np.logical_or(phasic>new_y,phasic<0)]=np.nan # phasic cannot be larger than raw data or negative

                myprint('...GSR data coverage %.1f%%' % (
                        100 * np.sum(~np.isnan(phasic)) / len(phasic)),file=report_file)

                signals['time ORIGINAL'] = signals['time']
                signals['time'] = new_time
                signals['responses'][r + ' ORIGINAL'] = signals['responses'][r]

                # force into positive values, again
                signals['responses'][r] = new_y
                signals['responses']['phasic'] = phasic
                signals['responses']['tonic'] = tonic #new_y - phasic

                #signals['responses']['hp_filtered'] = hp_filtered

                # make plot
                gsrfile = RESULT_FOLDER + file_name[:-4] + "_source%i_GSR" % (key_set+1)
                plt.close('all')
                plt.figure(figsize=[9,5],dpi=200)
                plt.plot(signals['time ORIGINAL'], signals['responses'][r + ' ORIGINAL'],'.',markersize=1)
                plt.plot(signals['time'], signals['responses']['phasic'],'.',markersize=1)
                plt.plot(signals['time'], signals['responses']['tonic'],'.',markersize=1)
                #plt.plot(signals['time'], signals['responses']['hp_filtered'],linewidth=1)
                plt.xlabel('Time [sec]')
                plt.ylabel('micro-Siemens')
                lgnd = plt.legend(['RAW', 'phasic', 'tonic'],loc=(1.02,0.50))# loc='center left')
                for i in range(3):
                    lgnd.legendHandles[i].update(props={'linestyle': '-'})

                #plt.legend(['RAW', 'phasic', 'tonic (res.)','%.3f-%.3fHz' % (BANDPASS_FILTER_RANGES[0], BANDPASS_FILTER_RANGES[1])],loc=(1.02, 0.50))  # loc='center left')
                add_annotations(plt.gca(), annotation_events)
                # plt.show()
                plt.gca().xaxis.set_minor_locator(tck.AutoMinorLocator())
                plt.title('data coverage %.1f%%' % (
                        100 * np.sum(~np.isnan(phasic)) / len(phasic)))
                plt.savefig(gsrfile+'.png',bbox_inches="tight")
                plt.savefig(gsrfile+'.pdf',bbox_inches="tight")
                # write data as text
                write_data(RESULT_FOLDER + file_name[:-4] + "_source%i_GSR.csv" % (key_set+1), signals, annotation_events,media_offset)

            if key == 'FACE':
                facefile = RESULT_FOLDER + file_name[:-4] + "_source%i_FACE" % (key_set+1)

                # make plot
                plt.close('all')
                fig = plt.figure()
                DPI = float(fig.get_dpi())
                coverage = np.nan
                try:
                    coverage = 100 * np.sum(signals['responses']['Number of faces'] > 0) / len(signals['responses']['Number of faces'])
                except:
                    pass
                myprint('...Affdex data coverage %.1f%%' % (coverage),file=report_file)
                fig.set_size_inches(700.0 / DPI, 1080.0 / DPI)
                for i, resp_key in enumerate(signals['responses'].keys()):
                    ax = plt.subplot(len(signals['responses']), 1, i + 1)
                    t = signals['time']
                    y = signals['responses'][resp_key]
                    plt.plot(t, y)
                    plt.ylabel(resp_key.split('|')[-1], rotation=0, horizontalalignment='right')
                    plt.xlim((t[0] - 1, t[-1] + 1))  # over timerange
                    if 'Number of' not in resp_key:
                        if "valence" in resp_key.lower():
                            ax.set_yticks([-100, 100], )
                            plt.ylim((-101, 101))  # results come scaled between -100-100
                            # ax.set_yticklabels([-100,100])
                            assert np.nanmin(y) > -100.01 and np.nanmax(
                                y) < 100.01, "AFFDEX data for %s not between [-100,100]!!!" % resp_key
                        else:
                            ax.set_yticks([0, 100])
                            plt.ylim((-1, 101))  # results come scaled between 0-100
                            assert np.nanmin(y) > -0.01 and np.nanmax(
                                y) < 100.01, "AFFDEX data for %s not between [0,100]!!!" % resp_key
                    if i + 1 < len(signals['responses']):
                        ax.set_xticklabels([])
                        add_annotations(plt.gca(), annotation_events, add_label=False)
                    else:
                        add_annotations(plt.gca(), annotation_events, add_label=True)
                        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
                    ax.tick_params(labelsize=7)
                plt.xlabel('Time [sec]')
                # plt.show(block=False)
                plt.subplots_adjust(left=0.25, bottom=0.10, right=0.98, top=0.98, wspace=0.05, hspace=0.09)
                plt.savefig(facefile+'.png',bbox_inches="tight")
                plt.savefig(facefile+'.pdf', bbox_inches="tight")
                # write data as text
                write_data(RESULT_FOLDER + file_name[:-4] + "_source%i_FACE.csv" % (key_set+1), signals, annotation_events,media_offset)

            if key == 'HEART':
                heartfile = RESULT_FOLDER + file_name[:-4] + "_source%i_HEART" % (key_set+1)
                t = signals['time']
                r = list(signals['responses'].keys())[0]  # get whatever response name
                y = signals['responses'][r]
                y[np.logical_or(y < 30, y > 230)] = np.nan  # assume valid pulse ranges

                # make plot
                plt.close('all')
                plt.figure(figsize=[9, 5], dpi=200)
                plt.plot(t, y,'-') #,markersize=3)
                plt.xlim((t[0] - 1, t[-1] + 1))  # over timerange
                plt.xlabel('Time [sec]')
                plt.ylabel('Heart rate [beats/min]')
                plt.title('data coverage %.1f%% (median %.1f)' % (
                (100 * np.sum(np.isnan(y) == 0) / len(y), np.nanmedian(y))))
                add_annotations(plt.gca(), annotation_events)
                plt.gca().xaxis.set_minor_locator(tck.AutoMinorLocator())
                plt.savefig(heartfile+'.png',bbox_inches="tight")
                plt.savefig(heartfile+'.pdf', bbox_inches="tight")
                myprint('...heartrate data coverage %.1f%% (median %.1f)' % (
                (100 * np.sum(np.isnan(y) == 0) / len(y), np.nanmedian(y))),file=report_file)
                write_data(RESULT_FOLDER + file_name[:-4] + "_source%i_HEART.csv" % (key_set+1), signals,annotation_events,media_offset)

            if key == 'EYE':
                xkey = [x for x in signals['responses'].keys() if 'X' in x][0]
                ykey = [x for x in signals['responses'].keys() if 'Y' in x][0]
                x_pos = signals['responses'][xkey]
                y_pos = signals['responses'][ykey]
                z = [x for x in signals['responses'].keys() if x not in [xkey,ykey]][0]
                fix_id = signals['responses'][z]
                mean_fixation = (np.nanmedian(x_pos), np.nanmedian(y_pos))
                myprint('...fixation coverage %.1f%% with median location x=%i and y=%i' % (
                100 * np.sum(np.isnan(fix_id) == 0) / len(fix_id), mean_fixation[0], mean_fixation[1]),file=report_file)
                # make plot
                plt.close('all')
                plot_eye_data(signals['time'], fix_id, x_pos, y_pos,RESULT_FOLDER + file_name[:-4] + "_source%i" % (key_set+1))
                write_data(RESULT_FOLDER + file_name[:-4] + "_source%i_EYE.csv" % (key_set+1), signals, annotation_events,media_offset)

    myprint('\nAll done! Result files written to path:\n  "%s"' % RESULT_FOLDER,file=report_file)

    report_file.close()

    if PROFILE:
        pr.disable()
        # after your program ends
        pr.print_stats(sort="cumtime")

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    Saleslab_settings.init() # read predefined params
    Saleslab_settings.process_file=lambda:process_file(Saleslab_settings.DATA_TO_EXTRACT)
    Saleslab_settings.myprint=myprint
    hasGUI = 0

    print('---- SalesLab raw data reader (%s, Janne Kauttonen) ----\nInfo: This script reads raw iMotions data, checks data validity and\nmakes preview images + cleaned datafiles for each modality it finds.\n' % Saleslab_settings.VERSION)
    if len(sys.argv)>1:
        print('Total %i arguments given' % (len(sys.argv)-1))
        INPUT_FILE = str(sys.argv[1])

    if INPUT_FILE is None:
        # no files, start GUI
        import Saleslab_GUI
        print('No input files given as arguments, starting GUI...\n')
        hasGUI=1
        Saleslab_GUI.start_GUI("Saleslab datareader")
    else:
        # we have file, process it and exit
        Saleslab_settings.INPUT_FILE = INPUT_FILE
        process_file(Saleslab_settings.DATA_TO_EXTRACT)

