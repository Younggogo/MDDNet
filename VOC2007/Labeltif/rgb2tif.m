clc;
clear all;
pathname=uigetdir(cd,'��ѡ���ļ���');
if pathname==0
    msgbox('��û����ȷѡ���ļ���');
    return;
end
filesjpg=ls(strcat(pathname,'\*.jpg'));
files=[cellstr(filesjpg)];
len=length(files);
TrainXraw = [];
TrainY = [];
for i = 1:len
    ImageName = files{i};
    rawfilename0 =strcat(pathname, '\', ImageName);
    ImageRaw = imread(rawfilename0);
    if length(ImageName) >= 13
        imagename = strcat(ImageName(1:9),'.tif');
        imwrite(ImageRaw,imagename);
    elseif length(ImageName) >= 5
        imagename = strcat(ImageName(1),'.tif');
        imwrite(ImageRaw,imagename); 
    else
        imagename = strcat(ImageName(2),'.tif');
        imwrite(ImageRaw,imagename); 
    end
end