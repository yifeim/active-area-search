function [handle] = clabel_precision(C,h,format)

handle=clabel(C,h,'fontsize',12);
for a=1:length(handle)
    s = get(handle(a),'String'); % get string
    s = str2num(s); % convert in to number
    s = sprintf(format,s); % format as you need
    set(handle(a),'String',s); % place it back in the figure
end
