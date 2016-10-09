classdef ActiveLevelSetEstimation < ActiveGP
  properties
    level
    eps_band
    beta_t

    pool_locs    % the pool of feasible locations
    C            % cumulatively shrinking confidence interval
    lse_outcome  % whether the value is above the level set or not
  end

  methods
    function self = ActiveLevelSetEstimation(gp_model, gp_para, level, beta_t, eps_band, pool_locs)

      self = self@ActiveGP(gp_model, gp_para);

      self.level     = level;
      self.beta_t    = beta_t;
      self.eps_band  = eps_band;

      self.pool_locs = pool_locs;

      n = size(pool_locs, 1);
      self.C           = [-inf(n, 1), inf(n, 1)];
      self.lse_outcome = zeros(n, 1);
    end

    function [new_lse_outcome] = update(self, locations, values)
      update@ActiveGP(self, locations, values);
      [new_lse_outcome] = self.update_lse();
    end

    function [new_lse_outcome] = update_lse(self)
      [~, ~, mu, var] = self.predict_points(self.pool_locs);
      std = sqrt(var);

      Q = [ ...
        mu - sqrt(self.beta_t) * std, ...
        mu + sqrt(self.beta_t) * std       ];

      % the cap operation
      self.C = [ ...
        max(Q(:,1), self.C(:,1)), ...
        min(Q(:,2), self.C(:,2))         ];

      new_lse_positive = (self.lse_outcome == 0) & (self.C(:,1) > self.level - self.eps_band);
      new_lse_negative = (self.lse_outcome == 0) & (self.C(:,2) < self.level + self.eps_band);

      new_lse_outcome = new_lse_positive - new_lse_negative;
      self.lse_outcome = self.lse_outcome + new_lse_outcome;
    end

    function [a] = utility(self)
      a = min(self.C(:,2) - self.level, self.level - self.C(:,1));
      a(self.lse_outcome ~= 0) = -inf;
    end
  end
end
