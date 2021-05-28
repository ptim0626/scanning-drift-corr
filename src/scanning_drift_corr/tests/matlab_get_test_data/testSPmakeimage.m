function [sMerge] = testSPmakeimage(scanAngles,varargin)

% Inputs:
% scanAngles            - Vector containg scan angles in degrees.
% images                - 2D image arrays, order in scanAngles, all same size.
% sigmaLP = 32; %       - sigma value in pixels for initial alignment LP filter. 
flagReportProgress = 1;  % Set to true to see updates on console.v =
paddingScale = (1+1/8);%1.5;    % - padding amount for scaling of the output.
sMerge.KDEsigma = 1/2; % - Smoothing between pixels for KDE.
sMerge.edgeWidth = 1/128; % - size of edge blending relative to input images.
sMerge.linearSearch = linspace(-0.02,0.02,1+2*2);  % Initial linear search vector, relative to image size.
% sMerge.linearSearch = linspace(-0.04,0.04,1+2*4);  % Initial linear search vector, relative to image size.

% flagSkipInitialAlignment = 0;  % Set to true to skip initial phase correlation.
% flagCrossCorrelation = 0+1;  % Set to true to use cross correlation.
% sigmaAlignLP = 64;  % low pass filtering for initial alignment (pixels).

% Initialize struct containing all data
sMerge.imageSize = round([size(varargin{1},1) size(varargin{1},2)]*paddingScale/4)*4;
sMerge.scanAngles = scanAngles;
sMerge.numImages = length(scanAngles);
sMerge.scanLines = zeros(size(varargin{1},1),size(varargin{1},2),sMerge.numImages);
sMerge.scanOr = zeros(size(varargin{1},1),2,sMerge.numImages);
sMerge.scanDir = zeros(sMerge.numImages,2);
sMerge.imageTransform = zeros(sMerge.imageSize(1),sMerge.imageSize(2),sMerge.numImages);
sMerge.imageDensity = zeros(sMerge.imageSize(1),sMerge.imageSize(2),sMerge.numImages);

% Generate all initial variables 
for a0 = 1:sMerge.numImages
    % Raw images
    if nargin == (sMerge.numImages+1)
        % Sequential images
        sMerge.scanLines(:,:,a0) = varargin{a0};
    elseif nargin == 2
        % 3D image stack
         sMerge.scanLines(:,:,a0) = varargin{1}(:,:,a0);
    else
        
    end
    
    % Generate pixel origins
    if nargin == (sMerge.numImages+1)
        xy = [(1:size(varargin{a0},1))' ones(size(varargin{a0},1),1)];
        xy(:,1) = xy(:,1) - size(varargin{a0},1)/2;
        xy(:,2) = xy(:,2) - size(varargin{a0},2)/2;
    elseif nargin == 2
        xy = [(1:size(varargin{1}(:,:,a0),1))' ones(size(varargin{1}(:,:,a0),1),1)];
        xy(:,1) = xy(:,1) - size(varargin{1}(:,:,a0),1)/2;
        xy(:,2) = xy(:,2) - size(varargin{1}(:,:,a0),2)/2;
    end
    
%     xy = [xy(:,1)*cos(scanAngles(a0)*pi/180) ...
%         - xy(:,2)*sin(scanAngles(a0)*pi/180) ...
%         xy(:,2)*cos(scanAngles(a0)*pi/180) ...
%         + xy(:,1)*sin(scanAngles(a0)*pi/180)];
    
    digitsOld = digits(50);
    D = vpa(deg2rad(scanAngles(a0)));
    xy = [xy(:,1)*cos(D) - xy(:,2)*sin(D) ...
          xy(:,2)*cos(D) + xy(:,1)*sin(D)];
    digits(digitsOld);
    xy = double(xy);
    
    xy(:,1) = xy(:,1) + sMerge.imageSize(1)/2;
    xy(:,2) = xy(:,2) + sMerge.imageSize(2)/2;
    

%     strange = xy(1,2);
%     save('num.mat', 'strange');

%     disp(33-xy(1,2));
    
    xy(:,1) = xy(:,1) - mod(xy(1,1),1);
    xy(:,2) = xy(:,2) - mod(xy(1,2),1);
    
%     disp(xy);
    
%     save("xy_init.mat", 'xy');    
    
    sMerge.scanOr(:,:,a0) = xy;

    % Scan direction
    sMerge.scanDir(a0,:) = [cos(scanAngles(a0)*pi/180 + pi/2) ...
        sin(scanAngles(a0)*pi/180 + pi/2)];
end

if flagReportProgress == true
    reverseStr = ''; % initialize console message piece
end




% First linear alignment, search over possible linear drift vectors.
sMerge.linearSearch = sMerge.linearSearch * size(sMerge.scanLines,1);
[yDrift,xDrift] = meshgrid(sMerge.linearSearch);
sMerge.linearSearchScore1 = zeros(length(sMerge.linearSearch));
inds = linspace(-0.5,0.5,size(sMerge.scanLines,1))';
N = size(sMerge.scanLines);
% w2 = circshift(padarray(hanningLocal(N(1))*hanningLocal(N(2))',...
%     sMerge.imageSize-N(1:2),0,'post'),round((sMerge.imageSize-N(1:2))/2));

% save('hanning_weights_small_delta.mat', 'w2');

% Calculate time dependent lienar drift
xyShift = [inds*xDrift(2,4) inds*yDrift(2,4)];

% disp(xyShift);

% Apply linear drift to first two images
sMerge.scanOr(:,:,1:2) = sMerge.scanOr(:,:,1:2) ...
                    + repmat(xyShift,[1 1 2]);
% Generate trial images
% sMerge = SPmakeImage(sMerge,1);
sMerge = SPmakeImage(sMerge,2);


end

