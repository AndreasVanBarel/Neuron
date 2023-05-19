# A simple line drawing program

module InputDraw 

export make_canvas
export get_lines
export get_line
export get_last_line

using Gtk.ShortNames
using GtkReactive
using Graphics 
using Colors

function make_canvas()
    win = Window("Drawing")  
    c = canvas(UserUnit)
    push!(win, c)

    lines = Signal([])   # the list of lines that we'll draw #const 
    newline = Signal([]) # the in-progress line (will be added to list above) #const 
    GtkReactive.gc_preserve(win, lines)
    GtkReactive.gc_preserve(win, newline)

    ##########

    drawing = Signal(false)  # this will become true if we're actively dragging #const 
    GtkReactive.gc_preserve(win, drawing)

    # c.mouse.buttonpress is a `Reactive.Signal` that updates whenever the
    # user clicks the mouse inside the canvas. The value of this signal is
    # a MouseButton which contains position and other information.

    # We're going to define a callback function that runs whenever the
    # button is clicked. If we just wanted to print the value of the
    # returned button object, we could just say
    #     map(println, c.mouse.buttonpress)
    # However, here our function is longer than `println`, so
    # we're going to use Julia's do-block syntax to define the function:
    sigstart = map(c.mouse.buttonpress) do btn
        # This is the beginning of the function body, operating on the argument `btn`
        if btn.button == 1 && btn.modifiers == 0 # is it the left button, and no shift/ctrl/alt keys pressed?
            push!(drawing, true)   # activate dragging
            push!(newline, [btn.position])  # initialize the line with the current position
        end
    end
    GtkReactive.gc_preserve(win, sigstart)

    ##########

    dummybutton = MouseButton{UserUnit}() #const 
    # See the Reactive.jl documentation for `filterwhen`
    sigextend = map(filterwhen(drawing, dummybutton, c.mouse.motion)) do btn
        # while dragging, extend `newline` with the most recent point
        push!(newline, push!(value(newline), btn.position))
    end
    GtkReactive.gc_preserve(win, sigextend)

    ##########

    sigend = map(c.mouse.buttonrelease) do btn
        if btn.button == 1
            push!(drawing, false)  # deactivate dragging
            # append our new line to the overall list
            push!(lines, push!(value(lines), value(newline)))
            # For the next click, make sure `newline` starts out empty
            push!(newline, [])
        end
    end
    GtkReactive.gc_preserve(win, sigend)

    ##########

    # Because `draw` isn't a one-line function, we again use do-block syntax:
    redraw = draw(c, lines, newline) do cnvs, lns, newl  # the function body takes 3 arguments
        fill!(cnvs, colorant"white")   # set the background to white
        set_coordinates(cnvs, BoundingBox(0, 1, 0, 1))  # set coordinates to 0..1 along each axis
        ctx = getgc(cnvs)   # gets the "graphics context" object (see Cairo/Gtk)
        for l in lns
            drawline(ctx, l, colorant"blue")  # draw old lines in blue
        end
        drawline(ctx, newl, colorant"red")    # draw new line in red
    end
    GtkReactive.gc_preserve(win, redraw)

    function drawline(ctx, l, color)
        isempty(l) && return
        p = first(l)
        move_to(ctx, p.x, p.y)
        set_source(ctx, color)
        for i = 2:length(l)
            p = l[i]
            line_to(ctx, p.x, p.y)
        end
        stroke(ctx)
    end

    Gtk.showall(win)

    return lines
    # todo: return canvas such that it can be closed
end 

function get_line(lines, index)
    # Get the last drawn line
    line = lines.value[index]

    line_values = Matrix{Float64}(undef,length(line),2)
    for i in 1:length(line)
        line_values[i,1] = line[i].x.val
        line_values[i,2] = line[i].y.val
    end
    return line_values
end

function get_lines(lines)
    [get_line(lines,i) for i in 1:length(lines.value)]
end 

get_last_line(lines) = get_line(lines, length(lines.value))


# Convert the drawn lines into an image matrix 
# We create a matrix of pixels
# The pixel values show the distance between the closest line segment 
# This is accomplished by going over all line segments and entering the distance in to it for each pixel in the matrix, updating pixels only if the new value is smaller than the old one 
# In this case, one starts out with a matrix filled with Infs

# Point is of the form [x,y]::Vector{Float64}
function add_point!(matrix, point)
    d(p1,p2) = sum((p1.-p2).^2)
    w,h = size(matrix)

    # produces coordinates in [0,1] × [0,1]
    to_unit_square(i,j) = [(i-1)/(w-1), (j-1)/(h-1)]

    distances = [d(point, to_unit_square(i,j)) for i in 1:w, j in 1:h]
    matrix.=min.(matrix, distances)
end

function add_line!(matrix, line)
    for i in 1:size(line,1)
        add_point!(matrix, line[i,:])
    end
end

function distance_matrix(line, w, h)
    matrix = fill(Inf32,w,h)
    add_line!(matrix, line)
    return sqrt.(matrix)
end



end
