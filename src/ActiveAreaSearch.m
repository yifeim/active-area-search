classdef ActiveAreaSearch < ActiveGP
  properties
  	regions     % g * 2d
  	level       % scalar
  	side        % scalar
  	highprob    % scalar
    cumfound       % g * 1
  end

  properties (Transient=true)
  	Z           % g * 1
  	omega       % g * n
    alpha       % g * 1, region posterior mean
    beta2       % g * 1, region posterior variance
    region_gpml_alpha   % g * 1, for expected reward computation
  end

  methods
    function self = ActiveAreaSearch(gp_model, gp_para, regions, level, side, highprob)

      self = self@ActiveGP(gp_model, gp_para)

      self.regions    = regions;
      self.level      = level;
      self.side       = side;
      self.highprob   = highprob;
      self.cumfound   = zeros(size(self.regions, 1), 1);

      [~, self.Z] = covSEregion(self.gp_para.cov, self.regions, []);

      self.alpha = 0 * self.Z;
      self.beta2 = self.Z;
    end

    function [Tg, new_found] = set(self, locations, values)
      self.set_points(locations, values)
      self.set_region_kernel(locations)
      self.compute_region_stats();
      [Tg, new_found] = self.region_rewards();
      self.cumfound = new_found;
    end

    function [Tg, new_found] = update(self, new_locs, new_vals)
      self.update_points(new_locs, new_vals);
      self.update_region_kernel(new_locs);
      self.compute_region_stats();
      [Tg, new_found] = self.region_rewards();
      self.cumfound = self.cumfound + new_found;
    end

    function [Tg, new_found] = region_rewards(self)
      Tg = normcdf(self.side .* (self.alpha - self.level) ./ sqrt(self.beta2));
      new_found = Tg > self.highprob;
    end 

    function [u, ug] = utility(self, pool_locs)
      ns = size(pool_locs, 1);
      g  = size(self.regions, 1);

      omegas = covSEregion(self.gp_para.cov, self.regions, pool_locs);
      [~, tV] = self.predict_points(pool_locs);

      if ~isempty(self.collected_locs)
        ks = self.gp_model.cov(self.gp_para.cov, self.collected_locs, pool_locs);
        omega_Vinv_ks = self.region_gpml_alpha' * ks;
      else
        omega_Vinv_ks = 0;
      end
      
      tnu = (abs(omegas - omega_Vinv_ks) ./ repmat(sqrt(tV)', g, 1));  % g * n
      tbeta2 = repmat(self.beta2, 1, ns) - tnu.^2;  % g * n

      assert(all(all(tbeta2>0)));

      alpham = repmat(self.alpha, 1, ns);

      ug = normcdf( (self.side .* (alpham - self.level) - sqrt(tbeta2).*norminv(self.highprob)) ./ tnu );

      u = sum(ug .* repmat(self.cumfound==0, 1, ns), 1);
    end
    
  end


  methods (Access = 'private')
    function [Tg, new_found] = compute_region_stats(self)
      % region mean integral is linear of queried data 
      assert(~isempty(self.collected_locs))

      self.alpha = self.omega * self.gp_post.alpha;

      V  = self.R' \ self.omega';
      self.beta2 = self.Z - sum(V.*V,1)';                       % predictive variances

      self.region_gpml_alpha = solve_chol(self.R, self.omega');

      assert(all(self.beta2>0));
    end

    function [] = set_region_kernel(self, locations)
      self.omega = covSEregion(self.gp_para.cov, self.regions, self.collected_locs);
    end

    function [] = update_region_kernel(self, new_locs)
      self.omega = [self.omega, covSEregion(self.gp_para.cov, self.regions, ...
        self.collected_locs((size(self.omega, 2)+1):end, :))];
    end
  end
end
