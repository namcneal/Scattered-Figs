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
				
		push!(layers, Dense(num_in, num_out, σ))
	end
	
	# This epsiode is brought to you by SquareShape
	square_shape = (word_sidelength, word_sidelength)
	
	# Add a final layer to shape it into a square
	push!(layers, x -> reshape(x, square_shape))
		
	return Chain(layers...)
end

# ╔═╡ bfed1d29-e521-4386-b926-edb20996f556
begin 
	word_sidelength = 5
	hidden_sizes    = [12, 24]
	speak = create_encoder(word_sidelength, hidden_sizes)
end

# ╔═╡ e5ab77d4-26c0-4c14-a8bf-c0d6fd7cb8c0
function visualise_speech(word::Array{Float64})
	matrix = Gray.(word)
	plot(matrix, xaxis=nothing, yaxis=nothing)
end

# ╔═╡ a2f03f3d-892c-4578-87d8-e93962652c08
visualise_speech(speak([0.1]))

# ╔═╡ Cell order:
# ╠═116dc840-b472-11eb-027b-a9f0194b81a1
# ╠═e2df7508-f930-47e5-9c7b-8c19d4b96457
# ╠═bfed1d29-e521-4386-b926-edb20996f556
# ╠═e5ab77d4-26c0-4c14-a8bf-c0d6fd7cb8c0
# ╠═a2f03f3d-892c-4578-87d8-e93962652c08
