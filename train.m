function W = train(vocab,cooccurrences,vector_size,iterations,learning_rate,x_max, alpha)

vocab_size = length(keys(vocab));
[i_main , i_context , cooccurrence] = find(cooccurrences);

%Word vector matrix. This matrix is (2V) * d, where N is the size
%of the corpus vocabulary and d is the dimensionality of the word
%vectors. All elements are initialized randomly in the range (-0.5,
%0.5]. We build two word vectors for each word: one for the word as
%the main (center) word and one for the word as a context word.
%It is up to the client to decide what to do with the resulting two
%vectors. Pennington et al. (2014) suggest adding or averaging the
%two for each word, or discarding the context vectors.
W = rand(2*vocab_size,vector_size) - 0.5;
%W =  (py.numpy.random.rand(vocab_size * 2, vector_size) - 0.5) / typecast(vector_size + 1,'double');


%Bias terms, each associated with a single vector. An array of size
%$2V$, initialized randomly in the range (-0.5, 0.5].
biases = rand(vocab_size * 2,1)-0.5;

gradient_squared = ones(vocab_size * 2, vector_size);

%Sum of squared gradients for the bias terms.
gradient_squared_biases = ones(vocab_size * 2,1);

%Training is done via adaptive gradient descent (AdaGrad). To make
%this work we need to store the sum of squares of all previous
%Like `W`, this matrix is (2V) * d.
%Initialize all squared gradient sums to 1 so that our initial

%Data Creation
data = py.list();

for i = 1:iterations
    
    global_cost = 0;
    
    for index = 1:length(i_main)
        % Error Here Not giving exact value
          v_main = W(i_main(index),:);
          v_context = W(i_context(index)+vocab_size,:);
          b_main = biases(i_main(index));
          b_context = biases(i_context(index)+vocab_size);
          gradsq_W_main = gradient_squared(i_main(index),:);
          gradsq_W_context = gradient_squared(i_context(index) + vocab_size,:);
          gradsq_b_main = gradient_squared_biases(i_main(index));
          gradsq_b_context = gradient_squared_biases(i_context(index)+vocab_size);
          cooccur = cooccurrence(index);

          if cooccur < x_max
                weight = (cooccur / x_max) ^ alpha;
          else
                weight = 1;
          end

    %Compute inner component of cost function, which is used in
    %both overall cost calculation and in gradient calculation
    %$$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$

          cost_inner = dot(v_main,v_context) + b_main + b_context - log(cooccur);

          %Compute cost
          % $$ J = f(X_{ij}) (J')^2 $$
          cost = weight * (cost_inner ^ 2);

          global_cost = global_cost + cost * 0.5;   

          %Compute gradients for word vector terms.
          %NB: `main_word` is only a view into `W` (not a copy), so our
          %modifications here will affect the global weight matrix;
          %likewise for context_word, biases, etc.
          grad_main = weight * cost_inner * v_context;
          grad_context = weight * cost_inner * v_main;

          %Compute gradients for bias terms
          grad_bias_main = weight * cost_inner;
          grad_bias_context = weight * cost_inner;

          %Now perform adaptive updates
          component_main = cellfun(@double,cell(py.list((learning_rate * grad_main / py.numpy.sqrt(py.list(gradsq_W_main))))));
          component_context = cellfun(@double,cell(py.list((learning_rate * grad_main / py.numpy.sqrt(py.list(gradsq_W_context))))));

          W(i_main(index),:) = W(i_main(index),:) - component_main; % actual update in W matrix, main_word
          W(i_context(index)+vocab_size,:) = W(i_context(index)+vocab_size,:) - component_context;

          biases(i_main(index))   = biases(i_main(index))-(learning_rate * grad_bias_main /sqrt(gradsq_b_main));
          biases(i_context(index)+vocab_size)   = biases(i_context(index)+vocab_size) - (learning_rate * grad_bias_context / sqrt(gradsq_b_context));


          %Update squared gradient sums
          gradient_squared(i_main(index),:)  = gradient_squared(i_main(index),:) + (grad_main.^2);
          gradient_squared(i_context(index) + vocab_size,:) =  gradient_squared(i_context(index) + vocab_size,:) + (grad_context.^2);
          gradient_squared_biases(i_main(index))    = gradient_squared_biases(i_main(index)) + grad_bias_main^2;
          gradient_squared_biases(i_context(index)+vocab_size) = gradient_squared_biases(i_context(index)+vocab_size) + grad_bias_context^2;
          %data.append({v_main,v_context,bias_main,bias_context,grad_main,grad_context,grad_bias_main,grad_bias_context,cooccur});

    end
    
     disp(global_cost);
    
end

end