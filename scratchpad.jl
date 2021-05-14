### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 116dc840-b472-11eb-027b-a9f0194b81a1
begin
	using Colors
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

# ╔═╡ 2003ee7c-825f-47d9-94ae-1f71b49ea08e
function create_decoder(word_sidelength::Int64, hidden_sizes::Vector{Int64})
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
	word_sidelength = 6
	hidden_sizes    = [32,64,32]
		
	init_speak  = create_encoder(word_sidelength, hidden_sizes)
	init_listen = create_decoder(word_sidelength, reverse(hidden_sizes))
end

# ╔═╡ e5ab77d4-26c0-4c14-a8bf-c0d6fd7cb8c0
function visualise_speech(word::Array{Float64})
	matrix = Gray.(word)
	p = plot(matrix, xaxis=nothing, yaxis=nothing)
	
	return p
end

# ╔═╡ a2f03f3d-892c-4578-87d8-e93962652c08
visualise_speech(init_speak([1.0]))

# ╔═╡ 7d6acb5a-d5a9-4c27-873a-fe23415252e1
begin
	num_words  = 10
	x = 1:num_words
	
	η = (y->y[1]) ∘ init_listen ∘ init_speak ∘ (x->[x])
	plot(x, η.(x), seriestype=:scatter, label="Autoencoder Output")
	plot!(x, x, ls=:dash, label="Desired Identity")
end

# ╔═╡ 2d6f656e-0d05-402f-9fd9-b13acb3516ab
begin
	speak  = deepcopy(init_speak)
	listen = deepcopy(init_listen)
	autoencoder = Chain(speak, listen)
	
	loss(x,y) = Flux.Losses.mse(autoencoder(x), y)
	optimiser = ADAM()
	θ = params(autoencoder)
		
	num_spoken = convert(Int, 1e4)
	words   = [[word] for word in rand(1.0:num_words, num_spoken)]
	dataset = zip(words, words)
	

	losses = []
	for iter in 1:5
		Flux.train!(loss, θ, dataset, optimiser)
		
		push!(losses, sum(loss.(words, words)))
	end
	
end

# ╔═╡ facdaa1c-f5f2-4234-a5d0-1ce9ac7ab328
begin
	# plot(losses)
	η_trained = (y->y[1]) ∘  autoencoder ∘ (x->[x])	
	plot(1:num_words, η_trained.(1:num_words), seriestype=:scatter, label="Autoencoder Output")
	plot!(1:num_words, 	1:num_words, ls=:dash, label="Desired Identity")

end

# ╔═╡ 67d5a26b-d25a-4969-bb5b-f484053d0c50
visualise_speech(speak([8.0]))

# ╔═╡ 943327fd-b6ca-4f8a-bed9-61129b78e017
visualise_speech(speak([9.0]))

# ╔═╡ ac7b78d0-fa52-42a2-84e6-e1a7fa0cd7cc
visualise_speech(speak([10.0]))

# ╔═╡ Cell order:
# ╠═116dc840-b472-11eb-027b-a9f0194b81a1
# ╠═e2df7508-f930-47e5-9c7b-8c19d4b96457
# ╠═2003ee7c-825f-47d9-94ae-1f71b49ea08e
# ╠═bfed1d29-e521-4386-b926-edb20996f556
# ╠═e5ab77d4-26c0-4c14-a8bf-c0d6fd7cb8c0
# ╠═a2f03f3d-892c-4578-87d8-e93962652c08
# ╠═7d6acb5a-d5a9-4c27-873a-fe23415252e1
# ╠═2d6f656e-0d05-402f-9fd9-b13acb3516ab
# ╠═facdaa1c-f5f2-4234-a5d0-1ce9ac7ab328
# ╠═67d5a26b-d25a-4969-bb5b-f484053d0c50
# ╠═943327fd-b6ca-4f8a-bed9-61129b78e017
# ╠═ac7b78d0-fa52-42a2-84e6-e1a7fa0cd7cc
