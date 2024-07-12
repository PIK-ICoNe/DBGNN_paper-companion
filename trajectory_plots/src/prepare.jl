include("utils.jl")

if layout_fontsize == "2images_per_column"
    guidefontsize=16
    tickfontsize=10
    legendfontsize=12
    right_margin = 20Plots.mm
    left_margin = 1Plots.mm
    bottom_margin = 1Plots.mm
elseif layout_fontsize == "3images_per_column"
    guidefontsize=24
    tickfontsize=15
    legendfontsize=18
    left_margin = 1.5Plots.mm
    right_margin = 22Plots.mm
    bottom_margin = 3Plots.mm
end
