classdef RandomSampling < ActiveGP
  methods
    function self = RandomSampling(gp_model, gp_para)
      self = self@ActiveGP(gp_model, gp_para);
    end

    function [u] = utility(self, pool_locs)
      u = rand(size(pool_locs, 1), 1);
    end
  end
end
