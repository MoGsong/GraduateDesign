%此文件为BC4_2a的初步预处理文件
%选取前22个通道
%制作CNT包
%后续的预处理使用opembmi工具箱的相关函数


function [EEG_MI_train,EEG_MI_test,Fs] = BCI4_2a_preprocess(i)
if nargin < 1
%     clear;clc;
    i = 3;
end


% %存储位置路径 更改

% 确定标签名称
marker={'1','left';'2','right';'3','foot';'4','tongue'};

%原始数据
% raw_pathT = sprintf('%s\\data\\BCIIV_2a\\BCICIV_2a_gdf\\A0%dT.gdf',cd, i);
% raw_pathE = sprintf('%s\\data\\BCIIV_2a\\BCICIV_2a_gdf\\A0%dE.gdf',cd, i);


path='C:\Users\Administrator\Desktop\GraduateDesign\BCI_data';

raw_pathT = sprintf('%s\\BCIIV_2a\\BCIIV_2a_mat\\A0%dT.mat',path, i);
raw_pathE = sprintf('%s\\BCIIV_2a\\BCIIV_2a_mat\\A0%dE.mat',path, i);
%测试集标签
class_pathE = sprintf('%s\\BCIIV_2a\\Data sets 2a_true_labels\\A0%dE.mat',path, i);
%为了确定标签名到导入的数据y_class
y_classpathT = sprintf('%s\\BCIIV_2a\\y_class\\A0%dT.xlsx',path, i);
y_classpathE = sprintf('%s\\BCIIV_2a\\y_class\\A0%dE.xlsx',path, i);

%确定电极名称


chan =   { 'Fz', 'FC3','FC1','FCz','FC2','FC4',...
           'C5', 'C3', 'C1','Cz','C52','C4','C6'...
           'CP3','CP1','CPz','CP2','CP4',...
           'P1','Pz','P2','POz'...
          }; 


for numpath=1:2
    if numpath == 1
        %读取数据

        load(raw_pathT);
        
        %选取前22个通道
        s = s(:,1:22);
        
        %使用中值滤波器处理数据  替换掉数据中的nan值   
        TF  = isnan(s);
        [II,JJ] = ind2sub(size(s),find(TF));
        s(II,JJ) = 0;            
        for kkkj = 1:length(II)
            mi = JJ(kkkj);
            mj = II(kkkj);
            if mj<=5
               s(mj,mi) = mean(s(1:10,mi));
            elseif mj>=size(s,1)-5
               s(mj,mi) = mean(s(end-9:end,mi));
            else
               s(mj,mi) = mean(s(mj-5:mj+5,mi)); 
            end
        end
        
        %频率
        Fs = h.SampleRate;
        
        %这里提取的时间点为cue点，提醒开始运动想象 hex2dec('301')=>769
        cue_pos_index  = ((h.EVENT.TYP == hex2dec('301')) + (h.EVENT.TYP == hex2dec('302'))+(h.EVENT.TYP == hex2dec('303'))+(h.EVENT.TYP == hex2dec('304')));
        cue_pos = h.EVENT.POS(cue_pos_index==1);
        
        tt = cue_pos';
        
        %确定训练集标签
        LABELS1 = h.Classlabel';

        % 确定y_class
        [NUM,TXT,RAW]=xlsread(y_classpathT);
        y_class = TXT';
        
        %制作CNT包
        EEG_MI_train.y_class = y_class;
        EEG_MI_train.t = tt;
        EEG_MI_train.fs = Fs;
        EEG_MI_train.y_dec = LABELS1;
        EEG_MI_train.y_logic = full(ind2vec(LABELS1));
        EEG_MI_train.class =marker;
        EEG_MI_train.x = s;
        EEG_MI_train.chan = chan;
        clear cue_pos;
    else
        %导入测试集数据
%         [s, h] = sload(raw_pathE);%, 0, 'OVERFLOWDETECTION:OFF'
        load(raw_pathE);
        
        %保留前22个通道
        s = s(:,1:22);
        
        %使用中值滤波器处理数据  替换掉数据中的nan值   
        TF  = isnan(s);
        [II,JJ] = ind2sub(size(s),find(TF));
        s(II,JJ) = 0;            
        for kkkj = 1:length(II)
            mi = JJ(kkkj);
            mj = II(kkkj);
            if mj<=5
               s(mj,mi) = mean(s(1:10,mi));
            elseif mj>=size(s,1)-5
               s(mj,mi) = mean(s(end-9:end,mi));
            else
               s(mj,mi) = mean(s(mj-5:mj+5,mi)); 
            end
        end
        
        Fs = h.SampleRate;
        
        %根据开始试验位置（300）来确定cue位置。  h.TRIG为实验开始时间，开始试验两秒后开始cue  
        cue_pos= h.TRIG'+ Fs*2;
                
        %导入标签数据
        load(class_pathE);
        
        LABELS2 = classlabel';

        %确定y_class
        [NUM,TXT,RAW]=xlsread(y_classpathE);
        y_class = TXT';
        
        %制作CNT包
        EEG_MI_test.y_class = y_class;
        EEG_MI_test.t = cue_pos;
        EEG_MI_test.fs = Fs;
        EEG_MI_test.y_dec = LABELS2;
        EEG_MI_test.y_logic = full(ind2vec(LABELS2));
        EEG_MI_test.class =marker;
        EEG_MI_test.x = s;
        EEG_MI_test.chan = chan;
%         save(pathall,'EEG_MI_train','EEG_MI_test')
    end
    
end
end