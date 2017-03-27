function global_cost = run_iter(vocab , data,learning_rate,x_max,alpha)
   
    global_cost = 0;
    py.random.shuffle(data);
    
    for dum = data
        
        v_main =  cellfun(@double,cell(py.list(dum{1}{1})));
        v_context = cellfun(@double,cell(py.list(dum{1}{2})));
        b_main =  dum{1}{3};
        b_context = dum{1}{4};
        gradsq_W_main =  cellfun(@double,cell(py.list(dum{1}{5})));
        gradsq_W_context = cellfun(@double,cell(py.list(dum{1}{6})));
        gradsq_b_main = dum{1}{7};
        gradsq_b_context = dum{1}{8};
        cooccurrence = dum{1}{9};
        
        if cooccurrence < x_max
            weight = (cooccurrence / x_max) ^ alpha;
        else
            weight = 1;
        end
        
%Compute inner component of cost function, which is used in
%both overall cost calculation and in gradient calculation
%$$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$

        cost_inner = dot(v_main,v_context) + b_main + b_context - log(cooccurrence);
        
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
        
        v_main    = v_main - component_main;
        v_context = v_context-component_context;        
        
        b_main   = b_main-(learning_rate * grad_bias_main /sqrt(gradsq_b_main));
        b_context = b_context-(learning_rate * grad_bias_context / sqrt(gradsq_b_context));

        %Update squared gradient sums
        gradsq_W_main  = gradsq_W_main + (grad_main.^2);
        gradsq_W_context =  gradsq_W_context+(grad_context.^2);
        gradsq_b_main    = gradsq_b_main+grad_bias_main^2;
        gradsq_b_context = gradsq_b_context+grad_bias_context^2;
        
    end 
end