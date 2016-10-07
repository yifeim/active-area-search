function [] = plot_sample(self, gp_model, gp_para, model_para,  regions, ...
  X1_view, X2_view, xqueried_aa, yqueried_aa, varargin) % blackdot, params

% parse optional inputs 
p = inputParser;
p.addOptional('blackdot', []);
p.addParamValue('clims', []);
p.addParamValue('clevels', []);
p.addParamValue('viz_valid', true(size(X1_view)));
p.addParamValue('verbose', false);
p.addParamValue('plotTailGaussian', false);
p.addParamValue('clabelOn', false);
p.addParamValue('labelOn', false); 
p.addParamValue('dotsize', 50); 
p.addParamValue('txtsize', []);
p.addParamValue('viz_m', []);
p.addParamValue('viz_ys2_D', []);
p.addParamValue('alpha', []);
p.addParamValue('beta2', []);



p.parse(varargin{:});

blackdot = p.Results.blackdot;
clims = p.Results.clims;
clevels = p.Results.clevels;
viz_valid = p.Results.viz_valid; 
verbose = p.Results.verbose; 
plotTailGaussian = p.Results.plotTailGaussian; 
clabelOn  = p.Results.clabelOn; 
labelOn = p.Results.labelOn; 
dotsize = p.Results.dotsize;
txtsize = p.Results.txtsize; 
viz_m = p.Results.viz_m;
viz_ys2_D = p.Results.viz_ys2_D; 
alpha = p.Results.alpha;
beta2 = p.Results.beta2; 


if isempty(txtsize), txtsize = dotsize/5; end

%/parse optional inputs

x_view = [X1_view(:), X2_view(:)];

granularity = regions{1}(2) - regions{1}(1);
siz_max = max(cell2mat(regions'),[],1); 
siz_min = min(cell2mat(regions'),[],1); 
siz = [siz_min(1), siz_max(2), siz_min(3), siz_max(4)];

ys2_prior = feval(gp_model.covfunc{:}, gp_para.cov, [nan nan]); 
if isempty(clims)
  % [~, Z0] = bmc_prior_box(regions, [], hyp);
  % Z0 = Z0(1);
  clims = [-2,2]*ys2_prior + model_para.levels;
end
if isempty(clevels)
  clevels = linspace(clims(1), clims(2), 5);
end


if isempty(viz_m) || isempty(viz_ys2_D)
  [viz_m, viz_ys2_D] = gpm(gp_model, gp_para, xqueried_aa, yqueried_aa, x_view);
end


% ------------------- common computations --------------------
% if isempty(alpha) || isempty(beta2)
%   [omega, Z] = bmc_prior_box_SE(regions, xqueried_aa, gp_para); 
%   K_queried = feval(gp_model.covfunc{:}, gp_para.cov, xqueried_aa);
%   invV_queried = cov4observed(K_queried, gp_model, gp_para);
%   
%   alpha     = omega * invV_queried * (yqueried_aa - model_para.mean(nan)) + model_para.mean(nan);
%   beta2     = Z - diag(omega * invV_queried * omega');
% end
% 
% curr_probs = normcdf(model_para.side * (alpha-model_para.levels) ./ sqrt(beta2));

curr_probs = self.model(xqueried_aa, yqueried_aa, regions, gp_model, gp_para, model_para);


if verbose
  fprintf( ['current probabilities: ', ...
    sprintf('%.3f ', curr_probs, abs(model_para.highprob)), ...
    '\n'] );
end

% ------------------- begin plotting -----------------------

image([siz(1), siz(2)], [siz(3),siz(4)], ...
  reshape(colorlookup(sqrt(max(viz_ys2_D,0)), [0, 3*sqrt(ys2_prior)], 1-gray), ...
  [size(X1_view),3] ));
set(gca,'yDir','normal');
hold on;

caxis(clims); colormap(brighten(jet, .7));
plot_patches(regions, 'rule'); hold on;

[c,h] = contour( X1_view,  X2_view,  reshape(viz_m, size(X1_view)).*viz_valid, ...
  clevels);
set(h,'linewidth',3);
if clabelOn
  h=clabel_precision(c,h,'%.2f'); 
  set(h,'fontsize', 20);
end

for i=1:length(regions)
  if plotTailGaussian
    plot_tail_gaussian([
      .5*regions{i}(1)+.5*regions{i}(2)
      .5*regions{i}(3)+.5*regions{i}(4)
      ], .2*granularity, model_para.levels,alpha(i), sqrt(beta2(i)), curr_probs(i)>abs(model_para.highprob), model_para.side);
  else
    plot_tick_cross_short([
      .5*regions{i}(1)+.5*regions{i}(2)
      .5*regions{i}(3)+.5*regions{i}(4)
      ], .2*granularity, curr_probs(i)>abs(model_para.highprob));
  end
end

if isempty(blackdot)
  scatter(xqueried_aa(:,1), xqueried_aa(:,2),dotsize,colorlookup(yqueried_aa),'filled');
  if labelOn
    for i=1:size(xqueried_aa,1)
      text(xqueried_aa(i,1), xqueried_aa(i,2),sprintf('%d',i),'fontsize',dotsize/5);
    end
  end
elseif ~strcmpi(blackdot{1}, 'null')
  scatter(xqueried_aa(:,1), xqueried_aa(:,2),dotsize,blackdot{:});
end


axis([siz(1) siz(2) siz(3) siz(4)]);
% drawnow;


function [h] = plot_tick_cross_short(offset, scale, reward)

h = [];

if reward == 1
  clr = 'r';
  x_shape = [-1  -.2  1];
  y_shape = [.2   -1   1];
else
  clr = 'g';%[.7 .7 .7];
  x_shape = [-1 -1; 1  1];
  y_shape = [-1  1; 1 -1];
end

x_target = offset(1) + .5*scale*x_shape;
y_target = offset(2) + .5*scale*y_shape; 

h = [];

h = [h, plot( x_target, y_target,'color',clr,'linewidth',1)];


function [rgb] = colorlookup(val, clim, cmap)

if nargin<2, clim = get(gca,'clim'); end

if nargin<3, cmap = colormap; end

ind = round( ...
  (val - clim(1)) / (clim(2)-clim(1)) * size(cmap,1) ...
  );

ind = max(ind, 1);
ind = min(ind, size(cmap,1));

rgb = cmap(ind, :);

