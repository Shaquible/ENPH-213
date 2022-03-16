function y = convolver(h, x)
%convolver give two arrays h and x to be convolved

    %resphing the arrays to convert any column vectors to row vectors
    h = reshape(h, 1, []);
    x = reshape(x, 1, []);
    %calculating number of non-zero elements in the convolution   
    N = length(h) + length(x) - 1;
    %filling the output array with zeros
    y = zeros(1, N);
    %resizing the input array with zeros to be the size length as y
    x(numel(y)) = 0;
    h(numel(y)) = 0;
    %calculating the convolution
    for n = 1:N
        i = 1;
        for k = n:-1:1
            y(n) = y(n) + x(k)*h(i);
            i = i + 1;    
        end
    end
    %plotting
    subplot(3,1,1)
    stem(x)
    title('x(n)')
    subplot(3,1,2)
    stem(h)
    title('h(n)')
    subplot(3,1,3)
    stem(y)
    title('y(n)')
end
