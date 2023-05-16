%���ļ�ΪBC4_2a�ĳ���Ԥ�����ļ�
%ѡȡǰ22��ͨ��
%����CNT��
%������Ԥ����ʹ��opembmi���������غ���


function [EEG_MI_train,EEG_MI_test,Fs] = BCI4_2a_preprocess(i)
if nargin < 1
%     clear;clc;
    i = 3;
end


% %�洢λ��·�� ����

% ȷ����ǩ����
marker={'1','left';'2','right';'3','foot';'4','tongue'};

%ԭʼ����
% raw_pathT = sprintf('%s\\data\\BCIIV_2a\\BCICIV_2a_gdf\\A0%dT.gdf',cd, i);
% raw_pathE = sprintf('%s\\data\\BCIIV_2a\\BCICIV_2a_gdf\\A0%dE.gdf',cd, i);


path='C:\Users\Administrator\Desktop\GraduateDesign\BCI_data';

raw_pathT = sprintf('%s\\BCIIV_2a\\BCIIV_2a_mat\\A0%dT.mat',path, i);
raw_pathE = sprintf('%s\\BCIIV_2a\\BCIIV_2a_mat\\A0%dE.mat',path, i);
%���Լ���ǩ
class_pathE = sprintf('%s\\BCIIV_2a\\Data sets 2a_true_labels\\A0%dE.mat',path, i);
%Ϊ��ȷ����ǩ�������������y_class
y_classpathT = sprintf('%s\\BCIIV_2a\\y_class\\A0%dT.xlsx',path, i);
y_classpathE = sprintf('%s\\BCIIV_2a\\y_class\\A0%dE.xlsx',path, i);

%ȷ���缫����


chan =   { 'Fz', 'FC3','FC1','FCz','FC2','FC4',...
           'C5', 'C3', 'C1','Cz','C52','C4','C6'...
           'CP3','CP1','CPz','CP2','CP4',...
           'P1','Pz','P2','POz'...
          }; 


for numpath=1:2
    if numpath == 1
        %��ȡ����

        load(raw_pathT);
        
        %ѡȡǰ22��ͨ��
        s = s(:,1:22);
        
        %ʹ����ֵ�˲�����������  �滻�������е�nanֵ   
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
        
        %Ƶ��
        Fs = h.SampleRate;
        
        %������ȡ��ʱ���Ϊcue�㣬���ѿ�ʼ�˶����� hex2dec('301')=>769
        cue_pos_index  = ((h.EVENT.TYP == hex2dec('301')) + (h.EVENT.TYP == hex2dec('302'))+(h.EVENT.TYP == hex2dec('303'))+(h.EVENT.TYP == hex2dec('304')));
        cue_pos = h.EVENT.POS(cue_pos_index==1);
        
        tt = cue_pos';
        
        %ȷ��ѵ������ǩ
        LABELS1 = h.Classlabel';

        % ȷ��y_class
        [NUM,TXT,RAW]=xlsread(y_classpathT);
        y_class = TXT';
        
        %����CNT��
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
        %������Լ�����
%         [s, h] = sload(raw_pathE);%, 0, 'OVERFLOWDETECTION:OFF'
        load(raw_pathE);
        
        %����ǰ22��ͨ��
        s = s(:,1:22);
        
        %ʹ����ֵ�˲�����������  �滻�������е�nanֵ   
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
        
        %���ݿ�ʼ����λ�ã�300����ȷ��cueλ�á�  h.TRIGΪʵ�鿪ʼʱ�䣬��ʼ���������ʼcue  
        cue_pos= h.TRIG'+ Fs*2;
                
        %�����ǩ����
        load(class_pathE);
        
        LABELS2 = classlabel';

        %ȷ��y_class
        [NUM,TXT,RAW]=xlsread(y_classpathE);
        y_class = TXT';
        
        %����CNT��
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