
clear
clc

classnum = 4;
% ���õ�����
reake = [];
BAND = [4,38];
fprintf('-----------------band: %d  \n',BAND);
class = ["left", "right", "foot", "tongue"];
% class_id �ֱ��Ӧ "l_r", "l_f", "l_t", "r_f", "r_t", "f_t"

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
%                 %�����ź�
% %                 traindata = [];
% %                 testdata = [];
% 
%                 [EEG_MI_train,EEG_MI_test,Fs] = BCI4_2a_preprocess(subject_id);
%                 EEG_train = prep_selectClass(EEG_MI_train,{'class',{    class(num(1)), class(num(2))   }});
%                 EEG_test = prep_selectClass(EEG_MI_test,{'class', {  class(num(1)), class(num(2))   }});
% 
%                 %���Ȱ���[-4.5 5.5]     [-1.5 5]�ֶ�
%                 ext_time = [-1.5 5];
%                 %��ȡѵ�����ݵ�ʱ��
%                 train_time = [0.5 2.5]; %[0.5 2.5];[0.3 2.3];
%                 %��ȡ�������ݵ�ʱ��  [-4 4]  
%                 test_time = [0.5 2.5]; %[0.5 2.5];[0.3 2.3]
% 
%                 exttime = [ext_time(1)*Fs*4, (ext_time(2)*Fs-1)*4];
%                 traintime = [(train_time(1))*Fs*4 ((train_time(2))*Fs-1)*4];
%                 testtime = [(test_time(1))*Fs*4 ((test_time(2))*Fs-1)*4];
%                  % ������ȡ
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
%                     % ά���û����������ͨ������
% %                     train_data = permute(train_data,[2,1,3]);
% %                     test_data = permute(test_data,[2,1,3]);
%                     % train_labels = permute(train_labels,[2,1]);
%                     % test_labels = permute(test_labels,[2,1]);
% 
%                     % ���ݷִ� ͨ��������/ƴ��
% %                     traindata = [traindata;train_data];
% %                     testdata = [testdata;test_data];
%                     
% %                 end
%                  % ά���û�����
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
% ������ǿ
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
                %�����ź�
                traindata = [];
                testdata = [];
                trainlabels = [];
                testlabels =[];

                [EEG_MI_train,EEG_MI_test,Fs] = BCI4_2a_preprocess(subject_id);
                EEG_train = prep_selectClass(EEG_MI_train,{'class',{    class(num(1)), class(num(2))   }});
                EEG_test = prep_selectClass(EEG_MI_test,{'class', {  class(num(1)), class(num(2))   }});

                %���Ȱ���[-4.5 5.5]     [-1.5 5]�ֶ�
                ext_time = [-1.5 5];
                %��ȡѵ�����ݵ�ʱ��
                train_time = [0.3 2.3]; %[0.5 2.5];[0.3 2.3];
                %��ȡ�������ݵ�ʱ��  [-4 4]  
                test_time = [0.3 2.3]; %[0.5 2.5];[0.3 2.3]

                exttime = [ext_time(1)*Fs*4, (ext_time(2)*Fs-1)*4];
                 % ������ȡ
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
                    % ά���û����������ͨ������
%                     train_data = permute(train_data,[2,1,3]);
%                     test_data = permute(test_data,[2,1,3]);
                    % train_labels = permute(train_labels,[2,1]);
                    % test_labels = permute(test_labels,[2,1]);

                    % ���ݷִ� ͨ��������/ƴ��
                    traindata = [traindata;train_data];
                    testdata = [testdata;test_data];
                    trainlabels = [trainlabels;train_labels];
                    testlabels =[testlabels;test_labels];
                end
                 % ά���û�����
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
% �ķ��� ��������������ǿ 
case 4
        for subject_id = 1:9

            bq=  ["l_r", "l_f", "l_t", "r_f", "r_t", "f_t"];
            fprintf('-----------------subject_id: %d  \n',subject_id);
    %             �����ź�
            traindata = [];
            testdata = [];
            trainlabels = [];
            testlabels =[];

            [EEG_MI_train,EEG_MI_test,Fs] = BCI4_2a_preprocess(subject_id);
            EEG_train = EEG_MI_train;%prep_selectClass(EEG_MI_train,{'class',{    class(num(1)), class(num(2))   }});
            EEG_test = EEG_MI_test;%prep_selectClass(EEG_MI_test,{'class', {  class(num(1)), class(num(2))   }});

            %             ���Ȱ���[-4.5 5.5]     [-1.5 5]�ֶ�
            ext_time = [-1.5 5];
            %             ��ȡѵ�����ݵ�ʱ�� ��[0.3 2.3]s,[0.4 2.4]s ����������ȡ����
            train_time = [0.3 2.3];
            %             ��ȡ�������ݵ�ʱ��  [-4 4]  [0.5 2.5]
            test_time = [0.3 2.3];
            %             ���ڻ������� step = 0.1s
            %             ѭ������ ��ȡ���� �ֱ𱣴�
            exttime = [ext_time(1)*Fs*4, (ext_time(2)*Fs-1)*4];
            for step = 0:0.1:0.4
                traintime = [(train_time(1)+step)*Fs*4 ((train_time(2)+step)*Fs-1)*4];
                testtime = [(test_time(1)+step)*Fs*4 ((test_time(2)+step)*Fs-1)*4];
                %       ������Ϊ[0.6 2.6]sʱ����ά�� X!=500  * 22 * 288�����������ݾ��ȶ�ʧ����
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
%                 % ά���û����������ͨ������
%                 train_data = permute(train_data,[2,1,3]);
%                 test_data = permute(test_data,[2,1,3]);
                % train_labels = permute(train_labels,[2,1]);
                % test_labels = permute(test_labels,[2,1]);
                
                % ���ݷִ� �����ϲ�
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
        %    �ķ�������Ԥ����    
%      case 4
%             for subject_id=remake
% 
%                 bq=  ["l_r", "l_f", "l_t", "r_f", "r_t", "f_t"];
%                 fprintf('-----------------subject_id: %d  \n',subject_id);
%                 %�����ź�
%                 traindata = [];
%                 testdata = [];
% 
%                 [EEG_MI_train,EEG_MI_test,Fs] = BCI4_2a_preprocess(subject_id);
%                 EEG_train = EEG_MI_train;%prep_selectClass(EEG_MI_train,{'class',{    class(num(1)), class(num(2))   }});
%                 EEG_test = EEG_MI_test;%prep_selectClass(EEG_MI_test,{'class', {  class(num(1)), class(num(2))   }});
% 
%                 %���Ȱ���[-4.5 5.5]     [-1.5 5]�ֶ�
%                 ext_time = [-1.5 5];
%                 %��ȡѵ�����ݵ�ʱ�� ��[0.3 2.3]s,[0.4 2.4]s ����������ȡ����
%                 train_time = [0.5 2.5];
%                 %��ȡ�������ݵ�ʱ��  [-4 4]  [0.5 2.5]
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
   
