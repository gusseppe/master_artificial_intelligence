
function [min_fval, min_x, min_params, record_avg] =  ...
    find_best_fit(initial_ranges, ...
                  population_list,  ...
                  generation_list,  ...
                  crossover_list, ...
                  ga_times)
              
             
    FitnessFunction = @(x)(1-x(1))^2+100*(x(2)-x(1)^2)^2; 
    
    % rng (random number generation) for reproducibility it takes the same random number
    rng default
    
    % Values to return
    min_fval = -1;
    min_x = 0;
    min_params = [];
    record_avg=[];
    
    % Run for all initial ranges
    for ran_i = 1:(size(initial_ranges, 1))
        
        % Run for all populations
        for pop_i = 1:(length(population_list))

            % Run for all generations
            for gen_i = 1:(length(generation_list))

                % Run for all 
                for cro_i = 1:(length(crossover_list))

                    % Set parameters
                    opts = gaoptimset('PopInitRange', initial_ranges(ran_i, :)');
                    opts = gaoptimset(opts,'PopulationSize', population_list(pop_i));
                    opts = gaoptimset(opts, 'Generations', generation_list(gen_i), 'Display', 'none');
                    opts = gaoptimset(opts,'CrossoverFraction', crossover_list(cro_i));

                    % Run ga_times to build mean value
                    record_fval=[];
                    for t=1:ga_times

                        % Run GA solver
                        [x, fval] = ga(FitnessFunction, 2, [], [], [], [], [], [], [], opts);
                        
                        % Collect values
                        record_fval = [record_fval; fval];
                        
                        % Assign minimal value if not set or currect is
                        % smaller than old minimal value
                        if (min_fval == -1) || (fval < min_fval)
                            min_fval = fval;
                            min_x = x;
                            min_params = [ ...
                                ran_i ...
                                pop_i ...
                                gen_i ...
                                cro_i ...
                            ];
                        end
                    end
                    
                    % Collect values as array
                    record_avg = [record_avg; mean(record_fval)];
                end
            end
        end
    end
end 


