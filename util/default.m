function default(varargin)

% single variable only, assign when ~exist or isempty

wb = 1;
wa = 1;
vstr = [varargin{:} ';'];
ii = find(vstr=='=');
vstr = [vstr(1:ii(1)) '1;'];
wb = who;
eval(vstr);
wa = who;
v = setdiff(wa,wb);

% str = [ '' ...
% 'if ~exist(''%s'',''var'') || isempty(%s),'...
% '  %s; '...
% 'end'];

if evalin('caller', sprintf('~exist(''%s'',''var'') || isempty(%s)', ...
    v{1}, v{1}))
  disp([varargin{:}]);
  evalin('caller', sprintf('  %s; ', [varargin{:}]));
else
  disp([v{1}, ' already defined']);
end

% str = sprintf(str,v{1},v{1}, [varargin{:}]);
% evalin('caller', str);

end

function []=disp(whatever)
end
