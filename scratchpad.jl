### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 116dc840-b472-11eb-027b-a9f0194b81a1
begin
	using Colors
	using DSP     # conv2
	using Flux 
	using Plots
end

# ╔═╡ e2df7508-f930-47e5-9c7b-8c19d4b96457
function create_encoder(word_sidelength::Int64, hidden_sizes::Vector{Int64})
	layers = []
	
	for h ∈ 0:length(hidden_sizes)
		if h == 0
			num_in  = 1
			num_out = hidden_sizes[1]
			σ       = relu
		elseif h == length(hidden_sizes)
			num_in  = hidden_sizes[end]
			num_out = word_sidelength^2
			σ       = sigmoid
		else
			num_in  = hidden_sizes[h]
			num_out = hidden_sizes[h+1]
			σ       = relu
		end
				
		push!(layers, Dense(num_in, num_out, σ; init=Flux.glorot_normal))
	end
	
	# This epsiode is brought to you by SquareShape
	square_shape = (word_sidelength, word_sidelength)
	
	# Add a final layer to shape it into a square
	push!(layers, x -> reshape(x, square_shape))
		
	return Chain(layers...)
end

# ╔═╡ e5ab77d4-26c0-4c14-a8bf-c0d6fd7cb8c0
function visualise_speech(word::Array{Float64})
	matrix = Gray.(word)
	return plot(matrix, xaxis=nothing, yaxis=nothing)
end

# ╔═╡ a2f03f3d-892c-4578-87d8-e93962652c08
visualise_speech(init_speak([1.0]))

# ╔═╡ 7d6acb5a-d5a9-4c27-873a-fe23415252e1
begin
	num_words  = 9
	x = 1:num_words
	
	η = (y->y[1]) ∘ init_listen ∘ init_speak ∘ (x->[x])
	plot(x, η.(x), seriestype=:scatter, label="Autoencoder Output")
	plot!(x, x, ls=:dash, label="Desired Identity")
end

# ╔═╡ c76240dc-6407-4938-b413-a80ec513d3c9
function similarity(arr1::Array{Float64}, arr2::Array{Float64},
	                       strictness::Float64)
	return exp(  -sum((arr1 .- arr2).^2) / strictness )
end
	

# ╔═╡ b04177ef-098c-411c-a099-a7dbae62630e
function local_average(array::Array{Float64}, kernel_size::Tuple{Int64, Int64})
	reshaped = reshape(array, (word_sidelength, word_sidelength, 1, 1))

	return MeanPool(kernel_size, stride=1)(reshaped)
end

# ╔═╡ 044dfcec-8b79-45ea-96c2-be78cbab2c7c
function local_similarity(arr1::Array{Float64}, arr2::Array{Float64},
	                     kernel_size::Tuple{Int64, Int64},
				         strictness::Float64)
	
	avg1 = local_average(arr1, kernel_size)
	avg2 = local_average(arr2, kernel_size)

	return similarity(avg1, avg2, strictness)
end

# ╔═╡ 2d6f656e-0d05-402f-9fd9-b13acb3516ab
begin
	speak  = deepcopy(init_speak)
	listen = deepcopy(init_listen)
	autoencoder = Chain(speak, listen)
	
	loss(x,y) = Flux.Losses.mse(autoencoder(x), y)
	
	function similarity_loss(x, y)
		return similarity(speak(x), speak(y), 10.0)
	end
		
	function local_similarity_loss(x, y)
		return local_similarity(speak(x), speak(y), (5,5), 20.0)
	end
	
	optimiser = ADAM()
	θ = params(autoencoder)
		
	num_spoken = convert(Int, 150*num_words)
	words   = rand(1.0:num_words, num_spoken, 2)
		
	num_iters = 50
	for _ in 1:num_iters
		for i in 1:num_spoken
			# # Update for understanding
			grads = gradient(() -> loss([words[i,1]], [words[i,1]]), θ)
			Flux.Optimise.update!(optimiser, θ, grads)
			
			# Update for clumping
			grads = gradient(() -> similarity_loss([words[i,1]], [words[i,2]]), θ)
			Flux.Optimise.update!(optimiser, θ, grads)
			
			# Update for clumping
		# 	grads = gradient(() -> local_similarity_loss([words[i,1]], [words[i,2]]), θ)
		# 	Flux.Optimise.update!(optimiser, θ, grads)
		end
	end

# 	losses = []
# 	for iter in 1:5
# 		Flux.train!(loss, θ, dataset, optimiser)
		
# 		Flux.train!(
		
# 		push!(losses, sum(loss.(words, words)))
# 	end
	
end

# ╔═╡ 8479e802-2251-4b46-a709-376f63d7695f
begin
	local_average(speak([9.0]), (3,3))
end

# ╔═╡ 2153e865-15c7-49d0-ace4-3f3cc4d3e9ec
local_similarity(speak([1.0]), speak([9.0]), (3,3), 3.0)

# ╔═╡ facdaa1c-f5f2-4234-a5d0-1ce9ac7ab328
begin
	# plot(losses)
	η_trained = (y->y[1]) ∘  autoencoder ∘ (x->[x])	
	plot(1:num_words, η_trained.(1:num_words), seriestype=:scatter, label="Autoencoder Output")
	plot!(1:num_words, 	1:num_words, ls=:dash, label="Desired Identity")

end

# ╔═╡ 53b431d2-2a5e-4a37-b3c5-9c73533d7f3f
begin
	vocabulary_plots = [visualise_speech(speak([x])) for x in 1.0:num_words]
	plot(vocabulary_plots..., layout=(3,3), framestyle = :box)
end

# ╔═╡ 2003ee7c-825f-47d9-94ae-1f71b49ea08e
function create_decoder(num_words::Int64, hidden_sizes::Vector{Int64})
	layers = []
	
	flat_shape = (word_sidelength^2, 1)
	
	# Take the square to a flat layer
	push!(layers, x -> reshape(x, flat_shape))
	
	for h ∈ 0:length(hidden_sizes)
		if h == length(hidden_sizes)
			num_in  = hidden_sizes[end]
			num_out = 1
			σ       = relu
		elseif h == 0
			num_in  = word_sidelength^2
			num_out = hidden_sizes[1]
			σ       = relu
		else
			num_in  = hidden_sizes[h]
			num_out = hidden_sizes[h+1]
			σ       = relu
		end
				
		push!(layers, Dense(num_in, num_out, σ; init=Flux.glorot_normal))
	end
	
	return Chain(layers...)
end

# ╔═╡ bfed1d29-e521-4386-b926-edb20996f556
begin 
	word_sidelength = 15
	hidden_sizes    = [25, 50]
		
	init_speak  = create_encoder(word_sidelength, hidden_sizes)
	init_listen = create_decoder(word_sidelength, reverse(hidden_sizes))
end

# ╔═╡ Cell order:
# ╠═116dc840-b472-11eb-027b-a9f0194b81a1
# ╠═e2df7508-f930-47e5-9c7b-8c19d4b96457
# ╠═2003ee7c-825f-47d9-94ae-1f71b49ea08e
# ╠═bfed1d29-e521-4386-b926-edb20996f556
# ╠═e5ab77d4-26c0-4c14-a8bf-c0d6fd7cb8c0
# ╠═a2f03f3d-892c-4578-87d8-e93962652c08
# ╠═7d6acb5a-d5a9-4c27-873a-fe23415252e1
# ╠═c76240dc-6407-4938-b413-a80ec513d3c9
# ╠═b04177ef-098c-411c-a099-a7dbae62630e
# ╠═8479e802-2251-4b46-a709-376f63d7695f
# ╠═044dfcec-8b79-45ea-96c2-be78cbab2c7c
# ╠═2153e865-15c7-49d0-ace4-3f3cc4d3e9ec
# ╠═2d6f656e-0d05-402f-9fd9-b13acb3516ab
# ╠═facdaa1c-f5f2-4234-a5d0-1ce9ac7ab328
# ╠═53b431d2-2a5e-4a37-b3c5-9c73533d7f3f
