function x = normalizeFeatures(x)
x = (x-min(x))./(max(x)-min(x));

end
