# Module supplying functionality for Decision Tree paper

module CellFateDecisionTrees

using DataFrames, DecisionTree, Colors

# Function to summarise the nodes of a decision tree in a dataframe
function summarise_tree(model, train_df)
    # Unpack full training dataframe
    labels = convert(Array{String}, train_df[:Classification])
    gene_names = string.(filter(x -> x != :Classification, names(train_df)))

    # Recurse through tree and write summary to dataframe
    df = sankey_recurse(model, train_df, gene_names, labels)[2]
    sort!(df, [:Depth])

    return df
end

# Function to get Sankey output text from a DecisionTree and write output to a
# destination text file. This output can be used to visualise models
# at http://sankeymatic.com/build/
function sankey_text(filedestination::String, model, train_df)

    labels = convert(Array{String}, train_df[:Classification])
    gene_names = string.(filter(x -> x != :Classification, names(train_df)))

    df = Inf
    open(filedestination, "w") do f
        node_color_runner = ""

        for unique_node in unique(labels)
            rand_col=RGB(rand(),rand(),rand())
            hexstring="#"*hex(rand_col)

            node_color_runner *= ":"*string(unique_node)*" "*hexstring*"\n"
        end
        (text, df) = sankey_recurse(model, train_df, gene_names, labels)[1:2]
        write(f, text*node_color_runner)
    end

    sort!(df, [:Depth])

    return df
end

# Recursive function that moves through the tree and assembles sankey output
function sankey_recurse(tree, train_df, gene_names, labels, runner = "",
     guage = 0, first_time = true, store = Dict(), parent_info = Dict(),
     df = DataFrame(Node_name = String[], Depth = Int[], Node_type = String[],
      Node_size = Int[], Parent_name = String[], Direction = String[],
      Entropy = Float64[], Samples = Any[]),
      index_samples = Int[0])

    # Get the name of the current node
    own_name = gene_names[tree.featid] * "," * string(round(tree.featval, digits=2))

    # If we are at the top of the tree, create an entry in the store dict to
    # record it's relationship to the root
    if first_time == true
        store[own_name] = ("Root", [])
        first_time = false
    end

    if guage == 0
        index_samples = collect(1:length(labels))
    end

    # Left samples
    l_samples = index_samples[train_df[index_samples,:][:,tree.featid].<=tree.featval]
    r_samples = index_samples[train_df[index_samples,:][:,tree.featid].> tree.featval]

    # Go left
    runner, store, parent_info, df, child_name_left = process_child(
     tree.left, train_df, own_name, gene_names, labels, runner, guage,
     first_time, store, parent_info, df, "L", l_samples)

    # Go right
    runner, store, parent_info, df, child_name_right = process_child(
     tree.right, train_df, own_name, gene_names, labels, runner, guage,
     first_time, store, parent_info, df, "R", r_samples)

    # Having considered both left and right, write the current and then delete
    if guage != 0
        SOURCE = store[own_name][1]
        TARGET = own_name
        AMOUNT = string(length(store[own_name][2]))

        runner *= SOURCE * " [" * AMOUNT *"] " * TARGET * "\n"
    end

    # Calculate entropy over labels at this node
    raw_ent = getEntropy(store[own_name][2])
    ent = raw_ent/length(unique(labels))

    # Convert entropy to color (0 = black, max = white) and add to runner
    col=RGB(ent,ent,ent)
    hexstring="#"*hex(col)
    runner *= ":"*own_name*" "*hexstring*"\n"

    # Remove this node's entry from the parent_info dict, as this node will not
    # be an antecendant to any additional nodes
    delete!(parent_info, own_name)

    # Append information about this node into the table
    push!(df, [own_name, guage, "Internal", length(index_samples), "Root",
     "na", ent, index_samples])

    # Check node-names for each of the child's names, and replace "parent_name"
    # category with "own_name"

    df[:Parent_name][df[:Node_name].== child_name_left] .= own_name
    df[:Direction][df[:Node_name].== child_name_left] .= "L"

    df[:Parent_name][df[:Node_name].== child_name_right] .= own_name
    df[:Direction][df[:Node_name].== child_name_right] .= "R"

    # Go back, up one level in the tree
    return runner, df, store, guage-1, first_time, parent_info

end

# Function to process a child::DecisionTree.Node
function process_child(child::DecisionTree.Node, train_df, parents_name,
     gene_names, labels, runner, guage, first_time, store, parent_info, df,
     direction, index_samples)

    # Add key for current node, if the dictionary doesn't already contain it
    if !(haskey(parent_info, parents_name))
        parent_info[parents_name] = []
    end

    # Get the name of the child node and create an entry in the store dict to
    # record its relationship to its parent
    child_name = gene_names[child.featid] * "," * string(round(child.featval, digits=2))
    store[child_name] = (parents_name, [])

    # Recurse on child node
    runner, df, store, guage, first_time, parent_info = sankey_recurse(
    child, train_df, gene_names, labels, runner, guage+1, first_time, store,
    parent_info, df, index_samples)

    return runner, store, parent_info, df, child_name
end

# Function to process a child::DecisionTree.Leaf
function process_child(child::DecisionTree.Leaf, train_df, parents_name,
     gene_names, labels, runner, guage, first_time, store, parent_info, df,
     direction, index_samples)

    # Add key for current node, if the dictionary doesn't already contain it
    if !(haskey(parent_info, parents_name))
        parent_info[parents_name] = []
    end

    # Get the labels contained within the child leaf node
    to_add = child.values

    # For every parent node for which this child is a descendant (i.e. those
    # nodes with a surviving entry in parent_info) add this child's labels
    # to the parent's entries in both dicts
    for (key, value) in parent_info
        parent_info[key] = vcat(parent_info[key], to_add)
        store[key] = (store[key][1], vcat(store[key][2], to_add))
    end

    # Also create entry for the current node
    SOURCE = parents_name
    TARGET = child.majority
    AMOUNT = string(length(to_add))

    # Submit entry by adding to runner
    runner *= SOURCE * " [" * AMOUNT *"] " * TARGET * "\n"

    # Calculate entropy at terminal node
    ent = getEntropy(to_add)

    # Append information about external node to dataframe too
    push!(df, [child.majority, guage+1, "External", length(index_samples),
     parents_name, direction, ent, index_samples])

    return runner, store, parent_info, df, "child_is_leaf"
end

# Function for calculating entropy over set of labels
function getEntropy(in_array)
    entropy = 0
    size = length(in_array)

    for class in unique(in_array)
        # px is the proportion of the number of elements in class x to th(e
        # number of elements in set S
        px = sum(in_array .== class) / size

        entropy += -px*log2(px)
    end

    return entropy
end

# Function for generating multiple decision trees and linking them together
# through a known hierarchy
function lineage_map(df, annotations, hierarchy, output_filename, tree_depth_limit = 0)
    nodes = Int.(unique(hierarchy[:Cluster_id]))

    # Create and store the models for each transition in the hierarchy tree view.
    # as well as array of strings for the cell labels encountered
    trees = Any[]
    labels_used = String[]

    gene_names = string.(filter(x -> x != :Classification, names(df)))
    sankey_runner = ""

    new_text = "blah"

    for current_node in nodes

        hierarchy_children = hierarchy[hierarchy[:Parent] .== current_node,:]

        # if node has no children, i.e. it is terminal on the tree_view
        # then just push "childless" to the trees array
        if size(hierarchy_children,1) == 0
            push!(trees, "childless")

        elseif size(hierarchy_children,1) == 1

            single_child_id = Int.(hierarchy_children[:Cluster_id])
            push!(trees, single_child_id)

            # Add the appropriate SOURCE - AMOUNT - TARGET info to runner text
            current_node_name = hierarchy[:Cluster_name][hierarchy[:Cluster_id] .== current_node][1]
            childs_name = hierarchy[:Cluster_name][hierarchy[:Cluster_id] .== single_child_id][1]

            # Set amount to the size of samples with childs_name as their label
            amount = string(sum(annotations[:Cluster_Label] .== childs_name))

            # Combine and append to full runner, for writing to text file later
            sankey_runner *= current_node_name * " [" * amount * "] " * childs_name * "\n"

        # Else the node has many children, and we must construct a decision tree
        # across these labels
        else
            # Get the names of the children nodes
            celltype_pool = hierarchy_children[:Cluster_name]

            # and use these children's names to find the gene expression entries for
            # all cells with these names
            df_children = df[[in(v, celltype_pool) for v in df[:Classification]],:]

            # Pull out the label and feature data, stored in type specific arrays
            labels = convert(Array{String}, df_children[:Classification])
            features = convert(Array{Float64}, df_children[filter(x -> x != :Classification, names(df))])

            # Learn the decision tree model to the correct depth
            model = build_tree(labels, features, 0, tree_depth_limit) # Limit to depth to 6
            labels_used = vcat(labels_used, unique(labels))

            # Push the resulting tree model to the trees array
            push!(trees, model)

            new_text = sankey_recurse(model, df, gene_names, labels)[1]

            # Connect new tree to existing portion
            # First, get the current node's name by accessing the hierarchy df
            current_node_name = hierarchy[:Cluster_name][hierarchy[:Cluster_id] .== current_node][1]

            # Second get the number of cells that have this label
            amount = string(length(labels))

            # Third, pull out name tree's top node from last entry of new_text
            # last_line = new_text[rsearchindex(new_text[1:end-1], "\n")+1:end]
            last_line = new_text[first(something(findlast("\n", new_text[1:end-1]), 0:-1))+1:end]
            # top_of_child = last_line[2:searchindex(last_line, "#")-2]
            top_of_child = last_line[2:first(something(findfirst("#", last_line), 0:-1))-2]

            # Combine and append to full runner, for writing to text file later
            to_add =  current_node_name * " [" * amount * "] " * top_of_child * "\n"
            #global sankey_runner *= to_add * new_text
            sankey_runner *= to_add * new_text
        end
    end



    # Generates random hexcodes to colour cell classes in sankey diagram. Append
    # this colour information to the sankey running text and save the output to
    # a .txt file to be visualised at http://sankeymatic.com/build/
    open(output_filename, "w") do f

        node_color_runner = ""
        for unique_node in unique(labels_used)
            rand_col=RGB(rand(),rand(),rand())
            hexstring="#"*hex(rand_col)

            node_color_runner *= ":"*string(unique_node)*" "*hexstring*"\n"
        end

        write(f, sankey_runner*node_color_runner)
    end

    return trees

end

# Umbrella function creating a list of all possible mutations a given sample
# could make for the model to re-classify it as the 'target_label'.
function generate_sample_mutations(data, model, sample, target_label)

    gene_list = names(data[filter(x -> x != :Classification, names(data))])

    # Get the sample's initial label
    og_label = apply_tree(model, sample)

    # Process and summarise tree structure in Dataframe
    model_summary = summarise_tree(model, data)

    # Generate a list of the node names
    node_names = model_summary[:Node_name]

    # Filter data for internal nodes, which are candidates for mutation
    internal_node_filter = model_summary[:Node_type].=="Internal"
    external_node_filter = model_summary[:Node_type].=="External"


    id_list = collect(1:size(model_summary,1))[internal_node_filter]
    internal_data = model_summary[internal_node_filter,:]

    # Describe the possible changes to the sample for each node in the treee
    changes = build_mutation_list(sample, internal_data, id_list, gene_list)

    # Identify the leaf nodes that would classify our sample as the target label
    external_node_id_list = collect(1:size(model_summary,1))[node_names .== target_label]

    # Find all unique paths through the tree that would generate this list
    lineage_list = get_lineage_traces(external_node_id_list, model_summary)

    # For each of these paths, return a list of nodes for which the sample would
    # have to be mutated
    nodes_to_mutate_across_lineages = get_required_mutations(lineage_list, model_summary, changes)

    # Convert this list of nodes to their proposed sample mutations
    proposed_mutations = generate_cohort(model, sample, changes, nodes_to_mutate_across_lineages)

    # Rank the proposed mutations by their summed absolute genetic distance
    abs_summed_distances = map(x -> sum(abs, x), proposed_mutations[:Distances])
    dist_rank = sortperm(abs_summed_distances)
    proposed_mutations = proposed_mutations[dist_rank, :]
    proposed_mutations[:Total_distance] = abs_summed_distances[dist_rank]

    return proposed_mutations
end

# Build a list of the individual mutations allowed by the model
function build_mutation_list(sample, internal_data, id_list, gene_list)

    possible_mutations = DataFrame(Node = Int64[], Gene = String[], Distance = Float64[],
     Gene_id = Int64[])

     for (ii, internal_node) in enumerate(internal_data[:Node_name])
         node_id = id_list[ii]

         gene, threshold = split(internal_node, ",")
         threshold = parse(Float64, threshold)

         gene_list_id = findfirst(isequal(Symbol(gene)), gene_list)

         sample_value = sample[gene_list_id]

         dist = threshold - sample_value

         push!(possible_mutations, [node_id gene dist gene_list_id])
     end

     return possible_mutations
end

# From the initial mutation list, generate and test all possible combinations
function generate_cohort(model, sample, change_list, nodes_to_mutate)

     track_changes = DataFrame(Factor = Int64[], Nodes = Array{Int64, 1}[],
     Genes = Array{String, 1}[], Gene_ids = Array{Int64, 1}[],
     Distances = Array{Float64, 1}[], Adjusted_label = String[], Altered_sample = Array{Float64,1}[])

     post_change_labels = []

     for combi in nodes_to_mutate

         altered_sample = deepcopy(sample)

         node_ids = []
         genes = []
         gene_ids = []
         distances = []

         for node_id in combi
             node_to_change_index = findfirst(isequal(node_id), change_list[:Node])
             row = change_list[node_to_change_index,:]

             altered_sample[row[:Gene_id]] = altered_sample[row[:Gene_id]] + (row[:Distance]*1.01)

             push!(node_ids, row[:Node])
             push!(genes, row[:Gene])
             push!(gene_ids, row[:Gene_id])
             push!(distances, row[:Distance])
         end

         factor = length(combi)

         new_label = apply_tree(model, altered_sample)

         push!(track_changes, (factor, node_ids, genes, gene_ids, distances, new_label, altered_sample))

     end

     return track_changes
end

# Given an array of starting node indices (relative to their rows in model_summary)
# return an array of arrays, the inner most of which traces the lineage up to the
# root node
function get_lineage_traces(list_of_starting_node_ids, model_summary)
    name_to_id = Dict(model_summary[:Node_name] .=> collect(1:size(model_summary,1)))
    external_node_filter = model_summary[:Node_type] .== "External"

    lineage_lists = []
    for starting_node_id in list_of_starting_node_ids
        curr_lineage = [starting_node_id]
        depth = model_summary[:Depth][starting_node_id]
        curr_node_id = starting_node_id

        while depth > 0
            parent_node_name = model_summary[:Parent_name][curr_node_id]
            parent_id = name_to_id[parent_node_name]

            push!(curr_lineage, parent_id)
            curr_node_id = parent_id

            depth = model_summary[:Depth][curr_node_id]
        end

        push!(lineage_lists, curr_lineage)
    end

    return lineage_lists
end

function get_required_mutations(lineage_list, model_summary, changes)
    nodes_to_mutate_across_lineages = []
    for lineage in lineage_list
        nodes_to_mutate = []

        for element_counter = 1:length(lineage)-1
            lineage_element = lineage[element_counter]
            lineage_direction = model_summary[lineage_element, :Direction]
            dist_to_parent_threshold = changes[:Distance][changes[:Node].==
             lineage[element_counter+1]]
            sample_direction = dist_to_parent_threshold... >= 0 ? "L" : "R"
            if sample_direction != lineage_direction
                push!(nodes_to_mutate, lineage[element_counter+1])
            end
        end

        push!(nodes_to_mutate_across_lineages, nodes_to_mutate)
    end
    return nodes_to_mutate_across_lineages
end

# Function that parses a DecisionTree object and returns the written rules
function get_rules(model, data, store = String[], rules = String[])
    if model isa Node
        name = string(names(data)[model.featid])
        thresh = string(round(model.featval, digits = 2))

        l_store = deepcopy(store)
        push!(l_store, "AND if " *name* " is less than or equal to " *thresh* " ")
        get_rules(model.left, data, l_store, rules)

        r_store = deepcopy(store)
        push!(r_store, "AND if " *name* " is greater than " *thresh* " ")
        get_rules(model.right, data, r_store, rules)
    else
        final = join(store) * "then sample belongs to class $(model.majority)"
        push!(rules, "I"*final[6:end])
    end

    return rules
end


end
