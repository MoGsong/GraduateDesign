clear all; clc; close all;

load('F:\BCI-data\BCIIV_2a\session5_data\A01E.mat');
CV.var.band=[7 20];
CV.var.interval=[750 3500];
CV.prep={ % commoly applied to training and test data before data split
    'CNT=prep_filter(CNT, {"frequency", band})'
    'SMT=prep_segmentation(CNT, {"interval", interval})'
    };
CV.train={
    '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
    };
CV.test={
    'SMT=func_projection(SMT, CSP_W)'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[cf_out]=func_predict(FT, CF_PARAM)'
    };
CV.option={
'KFold','7'
};

[loss]=eval_crossValidation(CNT, CV); 
















OpenBMI('E:\eeg\eeg_matlab_codes\USER_TOOLBOXS\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['C:\Users\Administrator\Desktop\BCI_Toolbox\DemoData'];

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

%% if you can redefine the marker information after Load_EEG function 
%% use  [marker, markerOrigin]=prep_defineClass(EEG.marke, marker)

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});

%% PRE-PROCESSING MODULE
band = [4 40];
channel_index = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6'];
time_interval =[1000 3500];
Niteration = 10;

% Pre-processing
for idx=1:2
    CNTch = prep_selectChannels(CNT{idx}, {'Char', channel_index});
    CNTchfilt =prep_filter(CNTch , {'frequency', band});
    all_SMT = prep_segmentation(CNTchfilt, {'interval', time_interval});
    if idx==1
        smt = all_SMT;
        clear all_SMT
    else
        SMT = prep_addTrials(smt, all_SMT);
    end
end

% Feature extracion and Classification
for iter=1:Niteration
    CV.train={
        '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", ["all"]})'...
        'FT=func_featureExtraction(SMT, {"feature","logvar"})'...
        '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'...
        };
    CV.test={
        'SMT=func_projection(SMT, CSP_W)'
        'FT=func_featureExtraction(SMT, {"feature","logvar"})'
        '[cf_out]=func_predict(FT, CF_PARAM)'
        };
    CV.option={
        'KFold','10'
        };
    [loss]=eval_crossValidation(SMT, CV);
    iter_result(1,iter)=1-loss;
end
Acc = mean(iter_result);
clear iter_result

