function [res] = get_option(opts, name, default)
    if isfield(opts, name)
        res = opts.(name);
    else
        res = default;
    end
end
