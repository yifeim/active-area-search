function [mn, so] = Gaussian_posterior( ...
    mode, mn0, so0, X, y, sigma_n)
  % function [mn, so] = Gaussian_posterior( ...
  %     mode, mn0, so0, X, y, sigma_n)
  % y = X*f = H' * f
  % log(y,f) = y'Xf - .5 f' X'X f - .5 (f-m0)' A0 (f-m0)
  % g(f) = X'y - X'Xf - A0(f-m0)
  % f = f0 + (A0+X'X) \ (X'y - X'Xf0 - A0(f0-m0))
  %   = (m0 + f0) + (A0+X'X) \ (X'y - X'X(f0+m0) - A0f0)
  %   = f0 + (A0+X'X) \ (X'y -X'Xf0 - A0f0 + A0m0)
  %   = (A0+X'X) \ (X'y + A0m0)
  
  if isscalar(y), y = y * ones(size(X,1),1); end;
  if isscalar(mn0), mn0 = mn0 * ones(size(X,2),1); end;

  assert( isempty(sigma_n) || ( length(sigma_n)==1 && sigma_n  >  0  ));
  if isempty(sigma_n); sigma_n = Inf; end
  %   y - X * p_vec0;
  
  assert( size(X, 1) == size(y, 1) );
  assert( size(X, 2) == size(mn0, 1)      );
  
  function [mn, prec] = posterior_prec( mn0, prec0 )
    prec = prec0 + sigma_n^-2 * (X' * X);
    mn = mn0 + prec \ (sigma_n^-2 * X'*(y-X*mn0));
  end
  
  function [mn, cov ] = posterior_cov ( mn0, cov0 )
    cov0_y = X * cov0 * X' + sigma_n^2 * eye(size(y, 1));
    R0_y = chol(cov0_y);
    minus_R = R0_y' \ (X * cov0);
    cov = cov0 - minus_R' * minus_R;
    mn = mn0 + cov  * (sigma_n^-2 * X'*(y-X*mn0));
  end
  
  function [mn, R   ] = posterior_chol( mn0, R0 )
    R0_s = R0*X';
    cov0_y =  R0_s'*R0_s + sigma_n^2 * eye(size(y, 1));
    R0_y = chol(cov0_y);
    minus_R = R0_y' \ R0_s' * R0;
    minus_L = minus_R';
    
    if size(minus_L, 2)<10
      R = full(R0);
      for okletscount = 1:size(minus_L, 2)
        R = cholupdate(R, minus_L(:, okletscount), '-');
      end
    else
      R = chol(R0'*R0 - minus_L * minus_L');
    end
    mn = mn0 + R'* (R * (sigma_n^-2 * X'*(y-X*mn0)));
    %     aa = [aa0 + R * (sigma_n^-2 * X'*(y-R0_s'*aa0)), ...
    %          R * (R0 \ aa0 + sigma_n^-2 * X'*y)];
  end

  switch mode
    case 'prec'
      [mn, so] = posterior_prec(mn0, so0);
    case 'cov'
      [mn, so] = posterior_cov (mn0, so0);
    case 'chol'
      [mn, so] = posterior_chol(mn0, so0);
  end
end

