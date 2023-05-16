
clear
clc

classnum = 4;
% 重置的数据
reake = [];
BAND = [4,38];
fprintf('-----------------band: %d  \n',BAND);
class = ["left", "right", "foot", "tongue"];
% class_id 分别对应 "l_r", "l_f", "l_t", "r_f", "r_t", "f_t"

switch classnum
%      case 2
%         for class_id = 1:6
%             for subject_id= remake
%                 if class_id==1
%                     num = [1, 2];
%                 elseif class_id==2
%                     num = [1, 3];
%                 elseif class_id==3
%                     num = [1, 4];
%                 elseif class_id==4
%                     num = [2, 3];
%                 elseif class_id==5
%                     num = [2, 4];
%                 elseif class_id==6
%                     num = [3, 4];
%                 end
%                 bq=  ["l_r", "l_f", "l_t", "r_f", "r_t", "f_t"];
% %                 fprintf('-----------------class: %s, subject_id: %d, band: %d-%d  \n',bq(class_id),subject_id,BAND(1),BAND(2));
%                 fprintf('-----------------class: %s, subject_id: %d  \n',bq(class_id),subject_id);
%                 %导入信号
% %                 traindata = [];
% %                 testdata = [];
% 
%                 [EEG_MI_train,EEG_MI_test,Fs] = BCI4_2a_preprocess(subject_id);
%                 EEG_train = prep_selectClass(EEG_MI_train,{'class',{    class(num(1)), class(num(2))   }});
%                 EEG_test = prep_selectClass(EEG_MI_test,{'class', {  class(num(1)), class(num(2))   }});
% 
%                 %首先按照[-4.5 5.5]     [-1.5 5]分段
%                 ext_time = [-1.5 5];
%                 %提取训练数据的时间
%                 train_time = [0.5 2.5]; %[0.5 2.5];[0.3 2.3];
%                 %提取测试数据的时间  [-4 4]  
%                 test_time = [0.5 2.5]; %[0.5 2.5];[0.3 2.3]
% 
%                 exttime = [ext_time(1)*Fs*4, (ext_time(2)*Fs-1)*4];
%                 traintime = [(train_time(1))*Fs*4 ((train_time(2))*Fs-1)*4];
%                 testtime = [(test_time(1))*Fs*4 ((test_time(2))*Fs-1)*4];
%                  % 滑窗截取
% %                 for step = 0:0.1:0.4
% %                     traintime = [(train_time(1)+step)*Fs*4 ((train_time(2)+step)*Fs-1)*4];
% %                     testtime = [(test_time(1)+step)*Fs*4 ((test_time(2)+step)*Fs-1)*4];
%                     % traintime = [train_time(1)*Fs*4 (train_time(2)*Fs-1)*4];
%                     % testtime = [test_time(1)*Fs*4 (test_time(2)*Fs-1)*4];
% %                     if step >= 0.3 && step < 0.4
% %                         traintime(2) =  traintime(2) + 8;
% %                         testtime(2) = testtime(2) + 8;
% %                     end
% %                     if  step >= 0.4
% %                         traintime(2) =  traintime(2) + 4;
% %                         testtime(2) = testtime(2) + 4;
% %                     end
%         %            EEG_train = prep_filter(EEG_train,{'frequency',BAND});
%                     eeg_train = prep_segmentation(EEG_train, {'interval', exttime});
%                     train = prep_selectTime(eeg_train,{'Time',traintime});
% 
%         %            EEG_test = prep_filter(EEG_test,{'frequency',BAND});   
%                     eeg_test = prep_segmentation(EEG_test, {'interval', exttime});
%                     test = prep_selectTime(eeg_test,{'Time',testtime});
% 
%                     train_data = permute(train.x,[2,3,1]);
%                     test_data = permute(test.x,[2,3,1]);
% 
%                     LABELS1 = train.y_dec';
%                     LABELS2 = test.y_dec';
% 
%                     fitst_label=LABELS1(1);
%                     [a , ~] =find(LABELS1(:) == fitst_label);
%                     [a1 , ~] =find(LABELS1(:) ~= fitst_label);
%                     [b , ~] =find(LABELS2(:) == fitst_label);
%                     [b1 , ~] =find(LABELS2(:) ~= fitst_label);
%                     LABELS1(a) = 1;
%                     LABELS1(a1) = 2;
%                     LABELS2(b) = 1;
%                     LABELS2(b1) = 2;
%     %                 train_labels= train.y_dec';
%     %                 test_labels= test.y_dec';
%                     train_labels = LABELS1;
%                     test_labels = LABELS2;
%                     % 维度置换，方便进行通道叠加
% %                     train_data = permute(train_data,[2,1,3]);
% %                     test_data = permute(test_data,[2,1,3]);
%                     % train_labels = permute(train_labels,[2,1]);
%                     % test_labels = permute(test_labels,[2,1]);
% 
%                     % 数据分窗 通道数叠加/拼接
% %                     traindata = [traindata;train_data];
% %                     testdata = [testdata;test_data];
%                     
% %                 end
%                  % 维度置换回来
% %                 train_data = permute(traindata,[2,1,3]);
% %                 test_data = permute(testdata,[2,1,3]);
%                 % BCI4_2a\\to_CNN_class2
%                 % SlideWindow_class2
%                 feapath = sprintf('%s\\data\\BCI4_2a\\to_CNN_class2\\%s',cd,bq(class_id));
%                 if exist(feapath,'dir') ==0
%                     mkdir(feapath);
%                 end
%                 data_feapath = sprintf('%s\\A0',feapath);
%                 save([data_feapath,num2str(subject_id),'.mat'] , 'train_data' , 'train_labels' , 'test_data' , 'test_labels')
%             end
%         end
% 数据增强
     case 2
        for class_id = 1:6
            for subject_id= 1:9
                if class_id==1
                    num = [1, 2];
                elseif class_id==2
                    num = [1, 3];
                elseif class_id==3
                    num = [1, 4];
                elseif class_id==4
                    num = [2, 3];
                elseif class_id==5
                    num = [2, 4];
                elseif class_id==6
                    num = [3, 4];
                end
                bq=  ["l_r", "l_f", "l_t", "r_f", "r_t", "f_t"];
%                 fprintf('-----------------class: %s, subject_id: %d, band: %d-%d  \n',bq(class_id),subject_id,BAND(1),BAND(2));
                fprintf('-----------------class: %s, subject_id: %d  \n',bq(class_id),subject_id);
                %导入信号
                traindata = [];
                testdata = [];
                trainlabels = [];
                testlabels =[];

                [EEG_MI_train,EEG_MI_test,Fs] = BCI4_2a_preprocess(subject_id);
                EEG_train = prep_selectClass(EEG_MI_train,{'class',{    class(num(1)), class(num(2))   }});
                EEG_test = prep_selectClass(EEG_MI_test,{'class', {  class(num(1)), class(num(2))   }});

                %首先按照[-4.5 5.5]     [-1.5 5]分段
                ext_time = [-1.5 5];
                %提取训练数据的时间
                train_time = [0.3 2.3]; %[0.5 2.5];[0.3 2.3];
                %提取测试数据的时间  [-4 4]  
                test_time = [0.3 2.3]; %[0.5 2.5];[0.3 2.3]

                exttime = [ext_time(1)*Fs*4, (ext_time(2)*Fs-1)*4];
                 % 滑窗截取
                for step = 0:0.1:0.4
                    traintime = [(train_time(1)+step)*Fs*4 ((train_time(2)+step)*Fs-1)*4];
                    testtime = [(test_time(1)+step)*Fs*4 ((test_time(2)+step)*Fs-1)*4];
                    traintime = [train_time(1)*Fs*4 (train_time(2)*Fs-1)*4];
                    testtime = [test_time(1)*Fs*4 (test_time(2)*Fs-1)*4];
                    if step >= 0.3 && step < 0.4
                        traintime(2) =  traintime(2) + 0;
                        testtime(2) = testtime(2) + 0;
                    end
                    if  step >= 0.4
                        traintime(2) =  traintime(2) + 0;
                        testtime(2) = testtime(2) + 0;
                    end
                   EEG_train = prep_filter(EEG_train,{'frequency',BAND});
                    eeg_train = prep_segmentation(EEG_train, {'interval', exttime});
                    train = prep_selectTime(eeg_train,{'Time',traintime});

        %            EEG_test = prep_filter(EEG_test,{'frequency',BAND});   
                    eeg_test = prep_segmentation(EEG_test, {'interval', exttime});
                    test = prep_selectTime(eeg_test,{'Time',testtime});

                    train_data = permute(train.x,[2,3,1]);
                    test_data = permute(test.x,[2,3,1]);

                    LABELS1 = train.y_dec';
                    LABELS2 = test.y_dec';

                    fitst_label=LABELS1(1);
                    [a , ~] =find(LABELS1(:) == fitst_label);
                    [a1 , ~] =find(LABELS1(:) ~= fitst_label);
                    [b , ~] =find(LABELS2(:) == fitst_label);
                    [b1 , ~] =find(LABELS2(:) ~= fitst_label);
                    LABELS1(a) = 1;
                    LABELS1(a1) = 2;
                    LABELS2(b) = 1;
                    LABELS2(b1) = 2;
    %                 train_labels= train.y_dec';
    %                 test_labels= test.y_dec';
                    train_labels = LABELS1;
                    test_labels = LABELS2;
                    % 维度置换，方便进行通道叠加
%                     train_data = permute(train_data,[2,1,3]);
%                     test_data = permute(test_data,[2,1,3]);
                    % train_labels = permute(train_labels,[2,1]);
                    % test_labels = permute(test_labels,[2,1]);

                    % 数据分窗 通道数叠加/拼接
                    traindata = [traindata;train_data];
                    testdata = [testdata;test_data];
                    trainlabels = [trainlabels;train_labels];
                    testlabels =[testlabels;test_labels];
                end
                 % 维度置换回来
                train_data = traindata;
                test_data =  testdata;
                train_labels = trainlabels;
                test_labels = testlabels;
                % BCI4_2a\\to_CNN_class2
                % SlideWindow_class2
                feapath = sprintf('%s\\data\\BCI4_2a\\SlideWindowDA_class2\\%s',cd,bq(class_id));
                if exist(feapath,'dir') ==0
                    mkdir(feapath);
                end
                data_feapath = sprintf('%s\\A0',feapath);
                save([data_feapath,num2str(subject_id),'.mat'] , 'train_data' , 'train_labels' , 'test_data' , 'test_labels')
            end
        end
% 四分类 滑动窗口数据增强 
case 4
        for subject_id = 1:9

            bq=  ["l_r", "l_f", "l_t", "r_f", "r_t", "f_t"];
            fprintf('-----------------subject_id: %d  \n',subject_id);
    %             导入信号
            traindata = [];
            testdata = [];
            trainlabels = [];
            testlabels =[];

            [EEG_MI_train,EEG_MI_test,Fs] = BCI4_2a_preprocess(subject_id);
            EEG_train = EEG_MI_train;%prep_selectClass(EEG_MI_train,{'class',{    class(num(1)), class(num(2))   }});
            EEG_test = EEG_MI_test;%prep_selectClass(EEG_MI_test,{'class', {  class(num(1)), class(num(2))   }});

            %             首先按照[-4.5 5.5]     [-1.5 5]分段
            ext_time = [-1.5 5];
            %             提取训练数据的时间 以[0.3 2.3]s,[0.4 2.4]s 滑动窗口提取数据
            train_time = [0.3 2.3];
            %             提取测试数据的时间  [-4 4]  [0.5 2.5]
            test_time = [0.3 2.3];
            %             窗口滑动步长 step = 0.1s
            %             循环滑动 提取数据 分别保存
            exttime = [ext_time(1)*Fs*4, (ext_time(2)*Fs-1)*4];
            for step = 0:0.1:0.4
                traintime = [(train_time(1)+step)*Fs*4 ((train_time(2)+step)*Fs-1)*4];
                testtime = [(test_time(1)+step)*Fs*4 ((test_time(2)+step)*Fs-1)*4];
                %       当窗口为[0.6 2.6]s时数据维度 X!=500  * 22 * 288，矩阵中数据精度丢失问题
                if step >= 0.3 && step < 0.4
                    traintime(2) =  traintime(2) + 8;
                    testtime(2) = testtime(2) + 8;
                end
                if  step >= 0.4
                    traintime(2) =  traintime(2) + 4;
                    testtime(2) = testtime(2) + 4;
                end
                
                eeg_train = prep_filter(EEG_train,{'frequency',BAND});
                eeg_train = prep_segmentation(eeg_train, {'interval', exttime});
                train = prep_selectTime(eeg_train,{'Time',traintime});

                eeg_test = prep_filter(EEG_test,{'frequency',BAND});   
                eeg_test = prep_segmentation(eeg_test, {'interval', exttime});
                test = prep_selectTime(eeg_test,{'Time',testtime});

                train_data = permute(train.x,[2,3,1]);
                test_data = permute(test.x,[2,3,1]);
                train_labels= train.y_dec';
                test_labels= test.y_dec';
                % train_labels = LABELS1;
                % test_labels = LABELS2;
%                 % 维度置换，方便进行通道叠加
%                 train_data = permute(train_data,[2,1,3]);
%                 test_data = permute(test_data,[2,1,3]);
                % train_labels = permute(train_labels,[2,1]);
                % test_labels = permute(test_labels,[2,1]);
                
                % 数据分窗 样本合并
                traindata = [traindata;train_data];
                testdata = [testdata;test_data];
                trainlabels = [trainlabels;train_labels];
                testlabels =[testlabels;test_labels];
                
            end

            train_data =traindata;
            test_data = testdata;
            train_labels = trainlabels;
            test_labels = testlabels;
            % to_CNN_class
            feapath = sprintf('%s\\data\\BCI4_2a\\SlideWindowDA_class%d',cd,classnum);
            if exist(feapath,'dir') ==0
                mkdir(feapath);
            end
            data_feapath = sprintf('%s\\A0',feapath);
            save([data_feapath,num2str(subject_id),'.mat'] , 'train_data' , 'train_labels' , 'test_data' , 'test_labels')


        end
        %    四分类数据预处理    
%      case 4
%             for subject_id=remake
% 
%                 bq=  ["l_r", "l_f", "l_t", "r_f", "r_t", "f_t"];
%                 fprintf('-----------------subject_id: %d  \n',subject_id);
%                 %导入信号
%                 traindata = [];
%                 testdata = [];
% 
%                 [EEG_MI_train,EEG_MI_test,Fs] = BCI4_2a_preprocess(subject_id);
%                 EEG_train = EEG_MI_train;%prep_selectClass(EEG_MI_train,{'class',{    class(num(1)), class(num(2))   }});
%                 EEG_test = EEG_MI_test;%prep_selectClass(EEG_MI_test,{'class', {  class(num(1)), class(num(2))   }});
% 
%                 %首先按照[-4.5 5.5]     [-1.5 5]分段
%                 ext_time = [-1.5 5];
%                 %提取训练数据的时间 以[0.3 2.3]s,[0.4 2.4]s 滑动窗口提取数据
%                 train_time = [0.5 2.5];
%                 %提取测试数据的时间  [-4 4]  [0.5 2.5]
%                 test_time = [0.5 2.5];
% 
%                 exttime = [ext_time(1)*Fs*4, (ext_time(2)*Fs-1)*4];
%                 traintime = [train_time(1)*Fs*4 (train_time(2)*Fs-1)*4];
%                 testtime = [test_time(1)*Fs*4 (test_time(2)*Fs-1)*4];
% 
%                 EEG_train = prep_filter(EEG_train,{'frequency',BAND});
%                 EEG_train = prep_segmentation(EEG_train, {'interval', exttime});
%                 train = prep_selectTime(EEG_train,{'Time',traintime});
% 
%                 EEG_test = prep_filter(EEG_test,{'frequency',BAND});   
%                 EEG_test = prep_segmentation(EEG_test, {'interval', exttime});
%                 test = prep_selectTime(EEG_test,{'Time',testtime});
% 
% 
%                 train_data = permute(train.x,[2,3,1]);
%                 test_data = permute(test.x,[2,3,1]);
%                 train_labels= train.y_dec';
%                 test_labels= test.y_dec';
%         %         train_labels = LABELS1;
%         %         test_labels = LABELS2;
% 
%                 feapath = sprintf('%s\\data\\BCI4_2a\\to_CNN_class%d',cd,classnum);
%                 if exist(feapath,'dir') ==0
%                     mkdir(feapath);
%                 end
%                 data_feapath = sprintf('%s\\A0',feapath);
%                 save([data_feapath,num2str(subject_id),'.mat'] , 'train_data' , 'train_labels' , 'test_data' , 'test_labels')
%             end
end
   
