% Running this file will download CIFAR10 and place the images into a
% training folder and test folder in the current directory
% These will be used for the three demos in this folder. 
% Please note this will take a few minutes to run, but only needs to be run
% once.
% Copyright 2017 The MathWorks, Inc.
%% Download the CIFAR-10 dataset
if ~exist('cifar-100-matlab','dir')
    cifar10Dataset = 'cifar-100-matlab';
    disp('Downloading CIFAR-100 dataset...');   
    websave([cifar10Dataset,'.tar.gz'],...
        ['https://www.cs.toronto.edu/~kriz/',cifar10Dataset,'.tar.gz']);
    gunzip([cifar10Dataset,'.tar.gz'])
    delete([cifar10Dataset,'.tar.gz'])
    untar([cifar10Dataset,'.tar'])
    delete([cifar10Dataset,'.tar'])
end    
   
%% Prepare the CIFAR-10 dataset
if ~exist('cifar100Train','dir')
    disp('Saving the Images in folders. This might take some time...');    
    saveCIFAR10AsFolderOfImages('cifar-100-matlab', pwd, true);
end


function saveCIFAR10AsFolderOfImages(inputPath, outputPath, varargin)
% saveCIFAR10AsFolderOfImages   Save the CIFAR-10 dataset as a folder of images
%   saveCIFAR10AsFolderOfImages(inputPath, outputPath) takes the CIFAR-10
%   dataset located at inputPath and saves it as a folder of images to the
%   directory outputPath. If inputPath or outputPath is an empty string, it
%   is assumed that the current folder should be used.
%
%   saveCIFAR10AsFolderOfImages(..., labelDirectories) will save the
%   CIFAR-10 data so that instances with the same label will be saved to
%   sub-directories with the name of that label.
% Check input directories are valid
if(~isempty(inputPath))
    assert(exist(inputPath,'dir') == 7);
end
if(~isempty(outputPath))
    assert(exist(outputPath,'dir') == 7);
end
% Check if we want to save each set with the same labels to its own
% directory.
if(isempty(varargin))
    labelDirectories = false;
else
    assert(nargin == 3);
    labelDirectories = varargin{1};
end
% Set names for directories
trainDirectoryName = 'cifar100Train';
testDirectoryName = 'cifar100Test';
% Create directories for the output
mkdir(fullfile(outputPath, trainDirectoryName));
mkdir(fullfile(outputPath, testDirectoryName));
if true
  clc;
  clear all;
  load('cifar-100-matlab/meta.mat');
  disp('loading meta...');    

  %Create folders
  for i=1:length(fine_label_names)
     mkdir('cifar100Test',fine_label_names{i});
     mkdir('cifar100Train',fine_label_names{i});
  end
  %%Training images
  load('cifar-100-matlab/train.mat');
  disp('loading train...');    

  im=zeros(32,32,3);
  for cpt=1:50000   
     R=data(cpt,1:1024);
     G=data(cpt,1025:2048);
     B=data(cpt,2049:3072);
     k=1;
     for x=1:32
        for i=1:32
          im(x,i,1)=R(k);
          im(x,i,2)=G(k);
          im(x,i,3)=B(k);
          k=k+1;
        end
     end  
     im=uint8(im);
     pathdest = strcat('cifar100Train/',fine_label_names{fine_labels(cpt)+1},'/',filenames{cpt});
     imwrite(im,pathdest,'png'); 
 end
 disp('finished train...');    

 %%Test images
 load('cifar-100-matlab/test.mat');
 disp('loading test...');
 im=zeros(32,32,3);
 for cpt=1:10000   
    R=data(cpt,1:1024);
    G=data(cpt,1025:2048);
    B=data(cpt,2049:3072);
    k=1;
    for x=1:32
       for i=1:32
          im(x,i,1)=R(k);
          im(x,i,2)=G(k);
          im(x,i,3)=B(k);
          k=k+1;
       end
    end  
    im=uint8(im);
    pathdest = strcat('cifar100Test/',fine_label_names{fine_labels(cpt)+1},'/',filenames{cpt});
    imwrite(im,pathdest,'png'); 
  end
end
disp('finished test...');    

end

% if(labelDirectories)
%     labelNames = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};
%     iMakeTheseDirectories(fullfile(outputPath, trainDirectoryName), labelNames);
%     iMakeTheseDirectories(fullfile(outputPath, testDirectoryName), labelNames);
%     for i = 1:5
%         iLoadBatchAndWriteAsImagesToLabelFolders(fullfile(inputPath,['data_batch_' num2str(i) '.mat']), fullfile(outputPath, trainDirectoryName), labelNames, (i-1)*10000);
%     end
%     iLoadBatchAndWriteAsImagesToLabelFolders(fullfile(inputPath,'test_batch.mat'), fullfile(outputPath, testDirectoryName), labelNames, 0);
% else
%     for i = 1:5
%         iLoadBatchAndWriteAsImages(fullfile(inputPath,['data_batch_' num2str(i) '.mat']), fullfile(outputPath, trainDirectoryName), (i-1)*10000);
%     end
%     iLoadBatchAndWriteAsImages(fullfile(inputPath,'test_batch.mat'), fullfile(outputPath, testDirectoryName), 0);
% end
% end
% function iLoadBatchAndWriteAsImagesToLabelFolders(fullInputBatchPath, fullOutputDirectoryPath, labelNames, nameIndexOffset)
% load(fullInputBatchPath);
% data = data'; %#ok<NODEF>
% data = reshape(data, 32,32,3,[]);
% data = permute(data, [2 1 3 4]);
% for i = 1:size(data,4)
%     imwrite(data(:,:,:,i), fullfile(fullOutputDirectoryPath, labelNames{labels(i)+1}, ['image' num2str(i + nameIndexOffset) '.png']));
% end
% end
% function iLoadBatchAndWriteAsImages(fullInputBatchPath, fullOutputDirectoryPath, nameIndexOffset)
% load(fullInputBatchPath);
% data = data'; %#ok<NODEF>
% data = reshape(data, 32,32,3,[]);
% data = permute(data, [2 1 3 4]);
% for i = 1:size(data,4)
%     imwrite(data(:,:,:,i), fullfile(fullOutputDirectoryPath, ['image' num2str(i + nameIndexOffset) '.png']));
% end
% end
% function iMakeTheseDirectories(outputPath, directoryNames)
% for i = 1:numel(directoryNames)
%     mkdir(fullfile(outputPath, directoryNames{i}));
% end
% end


% This function simply resizes the images to fit in AlexNet
% Copyright 2017 The MathWorks, Inc.
function I = readFunctionTrain(filename)
% Resize the images to the size required by the network.
I = imread(filename);
I = imresize(I, [227 227]);
end