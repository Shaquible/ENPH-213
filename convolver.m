function y = convolver(h, x)
%convolver give two arrays h and x to be convolved
    h = reshape(h, 1, []);
    x = reshape(x, 1, []);   
    N = length(h) + length(x) - 1;
    y = zeros(1, N);
    x(numel(y)) = 0;
    h(numel(y)) = 0;
    for n = 1:N
        i = 1;
        for k = n:-1:1
            y(n) = y(n) + x(k)*h(i);
            i = i + 1;    
        end
    end
    tiledlayout(3,1);
    ax1 = nexttile;
    title(x(n));
    ax2 = nexttile;
    ax3 = nexttile;
    stem(ax2, x);
    stem(ax1, h);
    stem(ax3, y);
end
