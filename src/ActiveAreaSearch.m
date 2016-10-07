classdef ActiveAreaSearch < handle
  properties
  	gp_model
  	gp_para
  	collected_locs  % n * d
  	collected_vals  % n * 1

  	regions     % g * 2d
  	level       % scalar
  	side        % scalar
  	highprob    % scalar
  end

  properties (Transient=true)
    gp_post
    R
    found       % g * 1
  	Z           % g * 1
  	omega       % g * n
    alpha       % g * 1, region posterior mean
    beta2       % g * 1, region posterior variance
    VinvOmega   % g * 1, for expected reward computation
    tV          % ns * 1
    tnu         % g * ns
    tbeta2      % g * ns
  end
  
  methods
    function self = ActiveAreaSearch(gp_model, gp_para, regions, level, side, highprob)

      assert( exp(2*gp_para.lik) > 1e-6 )  % gpml uses different representations otherwise

      self.gp_model   = gp_model;
      self.gp_para    = gp_para;
      self.regions    = regions;
      self.level      = level;
      self.side       = side;
      self.highprob   = highprob;

      % dummy data to avoid empty matrices
      self.set(1e100 * ones(1, size(self.regions, 2)/2), 0);
    end

    function [ymu, ys2, fmu, fs2] = predict_points(self, new_x)
      [ymu, ys2, fmu, fs2] = gp(self.gp_para, self.gp_model.inf, self.gp_model.mean, self.gp_model.cov, self.gp_model.lik, self.collected_locs, self.gp_post, new_x);
    end

    function [Tg, new_found] = set_region_rewards(self)
      % region mean integral is linear of queried data 
      nb_regions = size(self.omega, 1);
      self.alpha = self.omega * self.gp_post.alpha;

      V  = self.R' \ self.omega';
      self.beta2 = self.Z - sum(V.*V,1)';                       % predictive variances

      self.VinvOmega = solve_chol(self.R, self.omega');
      
      assert(all(self.beta2>0));
      
      % output is of dimension nb_groups*1
      Tg = normcdf(self.side .* (self.alpha - self.level) ./ sqrt(self.beta2));
      new_found = Tg > self.highprob;

    end

    function [Tg, new_found] = set(self, locations, values)
      self.collected_locs = locations;
      self.collected_vals = values(:);
      
      [~, ~, self.gp_post] = gp(self.gp_para, self.gp_model.inf, self.gp_model.mean, self.gp_model.cov, self.gp_model.lik, self.collected_locs, self.collected_vals);
      self.R = self.gp_post.L .* exp(self.gp_para.lik);

      if isempty(self.Z)
        [self.omega, self.Z] = covSEregion(self.gp_para.cov, self.regions, self.collected_locs);
      else
        self.omega = covSEregion(self.gp_para.cov, self.regions, self.collected_locs);
      end

      [Tg, new_found] = self.set_region_rewards();
      self.found = new_found;

    end


    function [Tg, new_found] = update(self, new_locs, new_vals)
      self.gp_post = update_posterior(self.gp_para, self.gp_model.mean, {self.gp_model.cov}, self.collected_locs, self.gp_post, new_locs, new_vals(:));
      self.R = self.gp_post.L .* exp(self.gp_para.lik);

      self.collected_locs = [self.collected_locs; new_locs];
      self.collected_vals = [self.collected_vals(:); new_vals(:)];

      self.omega = [self.omega, covSEregion(self.gp_para.cov, self.regions, ...
        self.collected_locs((size(self.omega, 2)+1):end, :))];

      [Tg, new_found] = self.set_region_rewards();
      self.found = self.found + new_found;

    end


    function [u, ug] = Ereward(self, pool_locs)
      % pool_locs is a p*m matrix, where p is the pool size. 
      % expected_rewards is a nb_regions*p vector 

      nb_evals = size(pool_locs, 1);
      nb_regions = size(self.regions, 1);
      
      omegas = covSEregion(self.gp_para.cov, self.regions, pool_locs);
      ks = self.gp_model.cov(self.gp_para.cov, self.collected_locs, pool_locs);

      [~, self.tV] = self.predict_points(pool_locs);
      
      self.tnu = (abs(omegas' - ks' * self.VinvOmega) ./ repmat(sqrt(self.tV), 1, nb_regions))';  % g * n
      self.tbeta2 = repmat(self.beta2, 1, nb_evals) - self.tnu.^2;  % g * n

      assert(all(all(self.tbeta2>0)));

      alpham = repmat(self.alpha, 1, nb_evals);

      ug = normcdf( (self.side .* (alpham - self.level) - sqrt(self.tbeta2).*norminv(self.highprob)) ./ self.tnu );

      u = sum(ug .* repmat(self.found==0, 1, nb_evals), 1);
    end
    
  end
end
