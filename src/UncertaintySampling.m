classdef UncertaintySampling < ActiveGP
  methods
    function self = UncertaintySampling(gp_model, gp_para)
      self = self@ActiveGP(gp_model, gp_para);
    end

    function [u] = utility(self, pool_locs)
      [~, u] = self.predict_points(pool_locs);
    end
  end
end
