from pymol import cgo, cmd
from typing import Literal
from collections.abc import Iterable
import numpy as np
import re

def _convert_color(color: str | tuple):
    """ Convert HEX code to RGB color """

    if color is None:
        return

    # If color is Hex code
    if isinstance(color, str):
        pattern = r"^#([A-Fa-f0-9]{3}|[A-Fa-f0-9]{4}|[A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$"
        if bool(re.match(pattern, color)):
            hex_color = color.lstrip('#')
            if len(hex_color) == 3:
                hex_color = ''.join(c*2 for c in hex_color)
            elif len(hex_color) == 4:
                hex_color = ''.join(c*2 for c in hex_color)
            if len(hex_color) == 6:
                return tuple(int(hex_color[i:i+2], 16)/255 for i in (0,2,4))
            elif len(hex_color) == 8:
                return tuple(int(hex_color[i:i+2], 16)/255 for i in (0,2,4,6))
            
        raise ValueError('Invalid HEX color code.')
        
    # If color is RGB in 255-scale
    elif len(color) >= 3 and np.any(np.array(color) > 1):
        return tuple(i/255 for i in color)
    elif len(color) >= 3 and np.all(np.array(color) <= 1):
        return tuple(color)
         
    
    raise ValueError('Invalid color.')

def _compute_normal(p1:tuple, p2:tuple, p3:tuple):
    """ Compute normal vector of a plane defined by 3 points """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p1) - np.array(p3)
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def _rotation(v1, v2):
    a = v1 / np.linalg.norm(v1)
    b = v2 / np.linalg.norm(v2)
    v = np.cross(a, b)
    c = np.dot(a, b)

    if c == 1:
        return np.identity(3)
    elif c == -1:
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        v = np.cross(a, perp)
        v = v / np.linalg.norm(v)
        return -np.identity(3) + 2 * np.outer(v, v)
    s = np.linalg.norm(v)
    kmat = np.array([[    0, -v[2],  v[1]],
                        [ v[2],     0, -v[0]],
                        [-v[1],  v[0],     0]])
    return np.identity(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))  

class PMShape:
    def __init__(self, name:str):
        
        self._name = str(name)
        self.graphic = []

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name:str):
        cmd.delete(self._name)
        self._name = str(name)
        self.set_graphic()
    
    def set_graphic(self):
        view = cmd.get_view() # get current view before update the object
        cmd.delete(self._name)
        cmd.load_cgo(self.graphic, self._name)
        cmd.set_view(view) # update view from the previous view
    
    def sphere(self, center:tuple, radius:float, color:str|tuple, alpha=1.0) -> list:
        x, y, z = center
        return [
            cgo.ALPHA, alpha,
            cgo.COLOR, *_convert_color(color), 
            cgo.SPHERE, float(x), float(y), float(z), float(radius),
        ]
    
    def line(self, start:tuple, end:tuple, linewidth:float, start_color:str|tuple, end_color:str|tuple) -> list:
        x1, y1, z1 = start
        x2, y2, z2 = end
        return [
            cgo.CYLINDER, 
            float(x1), float(y1), float(z1),
            float(x2), float(y2), float(z2),
            float(linewidth),
            *_convert_color(start_color), # start_color
            *_convert_color(end_color),   # end_color
        ]

    def triangle(self, p1, p2, p3, normal=None, color:str|tuple=(0.93, 0.93, 0.93), alpha=0.5):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3
        if not isinstance(normal, Iterable):
            normal = _compute_normal(p1, p2, p3)
        return [
            cgo.ALPHA, alpha,
            cgo.COLOR, *_convert_color(color),
            cgo.BEGIN, cgo.TRIANGLES,
            cgo.NORMAL, *normal,
            cgo.VERTEX, float(x1), float(y1), float(z1),
            cgo.VERTEX, float(x2), float(y2), float(z2),
            cgo.VERTEX, float(x3), float(y3), float(z3),
            cgo.END,
        ]

    def cone(self, base, tip, base_rad, base_color, tip_color):
        x1, y1, z1 = base
        x2, y2, z2 = tip
        return [
            cgo.CONE, 
            x1, y1, z1,
            x2, y2, z2,
            base_rad, 0,
            *_convert_color(base_color),
            *_convert_color(tip_color),
            1, 1
        ]

class Point(PMShape):
    def __init__(self, color:str|tuple, name:str):
        super().__init__(name)

        self._color = _convert_color(color)
    
    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, color:str|tuple):
        self._color = _convert_color(color)
        self.set_graphic()

class Patch(PMShape):
    def __init__(self, facecolor:str|tuple, edgecolor:str|tuple, edgewidth:float, 
                 normal:tuple=None, vertices:tuple=None, alpha=0.5, name='patch'):
        super().__init__(name)

        self._facecolor = _convert_color(facecolor)
        self._edgecolor = _convert_color(edgecolor)
        self._edgewidth = float(edgewidth)
        self._alpha = float(alpha)

        if self._alpha < 0 or self._alpha > 1:
            raise ValueError("'alpha' must be between 0 and 1 (with 0 meaning the object is invisible, and 1 is opaque).")

        if isinstance(vertices, Iterable):
            self._vertices = tuple(vertices)

        if isinstance(normal, Iterable):
            self._normal = np.array(normal)
            self._normal = self._normal / np.linalg.norm(self._normal)
            if len(self._normal) != 3:
                raise ValueError("'normal' must have length of 3.")
    
    @property
    def normal(self):
        return self._normal

    @normal.setter
    def normal(self, normal:tuple):
        self._normal = np.array(normal)
        self._normal = self._normal / np.linalg.norm(self._normal)
        if len(self._normal) == 3:
            return self._compute_vertices()
        raise ValueError("'normal' must have length of 3.")
    
    @property
    def vertices(self):
        raise NotImplementedError()

    @vertices.setter
    def vertices(self, vertices:Iterable):
        raise NotImplementedError()
    
    def _compute_vertices(self):
        raise NotImplementedError()

    @property
    def facecolor(self):
        return self._facecolor

    @facecolor.setter
    def facecolor(self, color:str|tuple):
        self._facecolor = _convert_color(color)
        self.set_graphic()
    
    @property
    def edgecolor(self):
        return self._edgecolor
    
    @edgecolor.setter
    def edgecolor(self, color:str|tuple):
        self._edgecolor = _convert_color(color)
        self.set_graphic()
    
    @property
    def edgewidth(self):
        return self._edgewidth
    
    @edgewidth.setter
    def edgewidth(self, edgewidth:float):
        self._edgewidth = float(edgewidth)
        self.set_graphic()

    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha:float):
        if self._alpha >= 0 and self._alpha <= 1:
            self._alpha = float(alpha)
            return self.set_graphic()

        raise ValueError("'alpha' must be between 0 and 1 (with 0 meaning the object is invisible, and 1 is opaque).")  

class Shape3D(PMShape):
    def __init__(self, center=(0,0,0), direction=(0,0,1),
                 width=5.0, height=5.0, depth=5.0, scale=1.0,
                 facecolor=(0.93,0.93,0.93), edgecolor=(1,0,0), 
                 edgewidth=0.02, alpha=0.5, name='shape3d'):
        super().__init__(name)

        self._height = float(height)
        self._width = float(width)
        self._depth = float(depth)
        self._scale = float(scale)
        self._center = np.array(center)
        self._direction = np.array(direction)
        self._facecolor = _convert_color(facecolor)
        self._edgecolor = _convert_color(edgecolor)
        self._edgewidth = float(edgewidth)
        self._alpha = float(alpha)

        if self._alpha < 0 or self._alpha > 1:
            raise ValueError("'alpha' must be between 0 and 1 (with 0 meaning the object is invisible, and 1 is opaque).")
    
    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center:tuple):
        self._center = np.asarray(center)
        if len(self._center) == 3:
            return self._compute_vertices()
        raise ValueError("'center' must have length of 3.")
    
    @property
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, direction:tuple):
        self._direction = np.array(direction)
        if len(self._direction) == 3:
            return self._compute_vertices()
        raise ValueError("'direction' must have length of 3.")
    
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, width:float):
        self._width = float(width)
        self._compute_vertices()
    
    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height:float):
        self._height = float(height)
        self._compute_vertices()
    
    @property
    def depth(self):
        return self._depth
    
    @depth.setter
    def depth(self, depth:float):
        self._depth = float(depth)
        self._compute_vertices()
    
    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale:float):
        self._scale = float(scale)
        self._compute_vertices()
    
    def _compute_vertices(self):
        raise NotImplementedError()

    @property
    def facecolor(self):
        return self._facecolor

    @facecolor.setter
    def facecolor(self, color:str|tuple):
        self._facecolor = _convert_color(color)
        self.set_graphic()
    
    @property
    def edgecolor(self):
        return self._edgecolor
    
    @edgecolor.setter
    def edgecolor(self, color:str|tuple):
        self._edgecolor = _convert_color(color)
        self.set_graphic()
    
    @property
    def edgewidth(self):
        return self._edgewidth
    
    @edgewidth.setter
    def edgewidth(self, edgewidth:float):
        self._edgewidth = float(edgewidth)
        self.set_graphic()

    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha:float):
        if self._alpha >= 0 and self._alpha <= 1:
            self._alpha = float(alpha)
            return self.set_graphic()

        raise ValueError("'alpha' must be between 0 and 1 (with 0 meaning the object is invisible, and 1 is opaque).")  

####################################

class Sphere(Point):
    def __init__(self, pos=[0,0,0], radius=0.1, color=(0.93,0.93,0.93), alpha=1.0, name='sphere'):
        
        """ 
        Draw a Sphere shape 
        
        Parameters
        ----------
        pos: list[float, float, float], default : [0,0,0]
            The Cartesian coordinates of the position of the sphere.
        radius: float, default : 0.1
            The radius of the sphere.
        color: str | tuple, default : (0.93, 0.93, 0.93)
            Color of the sphere, must be either HEX code or RGB color.
        alpha: float, default : 1.0
            Transparency of the sphere. `alpha` must be between 0 and 1
            (with 0 meaning the object is invisible, and 1 is opaque).
        name: str, default : 'sphere'
            Name of the sphere used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".
        
        """
        super().__init__(color, name)

        self._pos = tuple(pos)
        self._radius = float(radius)
        self._alpha = float(alpha)

        if len(self._pos) != 3:
            raise ValueError("'pos' must have length of 3.")
        if self._alpha < 0 or self._alpha > 1:
            raise ValueError("'alpha' must be between 0 and 1 (with 0 meaning the object is invisible, and 1 is opaque).")
        
        self.set_graphic()
    
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos:tuple):
        self._pos = tuple(pos)
        if len(self._pos) == 3:
            return self.set_graphic()
        raise ValueError("'pos' must have length of 3.")
    
    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius:float):
        self._radius = float(radius)
        self.set_graphic()
    
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha:float):
        if self._alpha >= 0 and self._alpha <= 1:
            self._alpha = float(alpha)
            return self.set_graphic()

        raise ValueError("'alpha' must be between 0 and 1 (with 0 meaning the object is invisible, and 1 is opaque).")  
    
    def set_graphic(self):
        self.graphic = self.sphere(self._pos, self._radius, self._color, self._alpha)
        return super().set_graphic()

class Line(Point):
    def __init__(self, start=[0,0,0], end=[0,0,1], color=(0.93,0.93,0.93), linewidth=0.05, 
                 capstyle:Literal['butt','round']='round', 
                 linestyle:Literal['solid','dash','dot']='solid',
                 arrow:Literal['none','both','end','start']='none',
                 gaplength=0.2, name='line'):

        """ 
        Draw a Line as a Cylinder shape 
        
        Parameters
        ----------
        start: list[float, float, float], default : [0,0,0]
            The Cartesian coordinates of the starting point of the line.
        end: list[float, float, float], default : [0,0,1]
            The Cartesian coordiantes of the end point of the line.
        color: str | tuple, default : (0.93, 0.93, 0.93)
            Color of the line, must be either HEX code or RGB color.
        linewidth: float, default : 0.02
            Width of the line.
        capstyle: Literal['butt', 'round'], default : 'round'
            Capstyle of the line. 
        linestyle: Literal['solid', 'dash', 'dot'], default : 'solid'
            Linestyle of the line.
        gaplength: float, default : 0.2
            If `linestyle` is 'dash' or 'dot', `gaplength` will determine space between dash/dot segments.
            Otherwise, `gaplength` will be ignored.
        arrow: Literal['none', 'both', 'end', 'start'], default : 'none'
            Whether to draw arrows.
        name: str, default : 'line'
            Name of the line used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".
        
        """
        super().__init__(color, name)
        
        self._start = np.array(start)
        self._end = np.array(end)
        self._linewidth = float(linewidth)
        self._capstyle = capstyle
        self._linestyle = linestyle
        self._gaplength = float(gaplength)
        self._arrow = arrow

        if len(self._start) != 3:
            raise ValueError("'start' must have length of 3.")
        if len(self._end) != 3:
            raise ValueError("'end' must have length of 3.")
        if self._capstyle not in ['butt', 'round']:
            raise ValueError("Invalid 'capstyle'. Available options for 'capstyle' are ['butt', 'round'].")
        if self._linestyle not in ['solid','dash','dot']:
            raise ValueError("Invalid 'linestyle'. Available options for 'linestyle' are ['solid','dash','dot'].")
        
        self.set_graphic()
    
    @property
    def start(self):
        return self._start
    
    @start.setter
    def start(self, pos:tuple):
        self._start = np.array(pos)
        if len(self._start) == 3:
            return self.set_graphic()
        raise ValueError("'start' must have length of 3.")
    
    @property
    def end(self):
        return self._end
    
    @end.setter
    def end(self, pos:tuple):
        self._end = np.array(pos)
        if len(self._end) == 3:
            return self.set_graphic()
        raise ValueError("'end' must have length of 3.")

    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, linewidth:float):
        self._linewidth = float(linewidth)
        self.set_graphic()

    @property
    def capstyle(self):
        return self._capstyle
    
    @capstyle.setter
    def capstyle(self, capstyle:Literal['butt','round']):
        if capstyle in ['butt', 'round']:
            self._capstyle = capstyle
            return self.set_graphic()
        
        raise ValueError("Invalid 'capstyle'. Available options for 'capstyle' are ['butt', 'round'].")
    
    @property
    def linestyle(self):
        return self._linestyle

    @linestyle.setter
    def linestyle(self, linestyle:Literal['solid','dash','dot']):
        if linestyle in ['solid','dash','dot']:
            self._linestyle = linestyle
            return self.set_graphic()
        
        raise ValueError("Invalid 'linestyle'. Available options for 'linestyle' are ['solid','dash','dot'].")
    
    @property
    def gaplength(self):
        return self._gaplength

    @gaplength.setter
    def gaplength(self, gaplength:float):
        self._gaplength = float(gaplength)
        self.set_graphic()
    
    @property
    def arrow(self):
        return self._arrow

    @arrow.setter
    def arrow(self, arrow:str):
        if arrow in ['none','both','end','start']:
            self._arrow = arrow
            return self.set_graphic()
        
        raise ValueError("Invalid 'arrow'. Available options for 'arrow' are ['none','both','end','start'].")

    def set_graphic(self):

        self.graphic = []

        if self._linestyle == 'solid':
            self.graphic = self.line(self._start, self._end, self._linewidth, self._color, self._color)
        
            if self._capstyle == 'round':
                self.graphic += self.sphere(self._start, self._linewidth, self._color, 1.0)
                self.graphic += self.sphere(self._end, self._linewidth, self._color, 1.0)
                
        elif self._linestyle == 'dash':
            pos = 0
            vec = self._end - self._start
            length = np.linalg.norm(vec)
            
            i = 1
            while True:
                dashlength = (length + self._gaplength) / i - self._gaplength
                if dashlength <= self._gaplength * 0.5:
                    break
                i += 1

            while pos < length:
                start = self._start + vec / length * pos
                end = self._start + vec / length * min(pos + dashlength, length)
                self.graphic += self.line(start, end, self._linewidth, self._color, self._color)

                if self._capstyle == 'round':
                    self.graphic += self.sphere(start, self._linewidth, self._color, 1.0)
                    self.graphic += self.sphere(end, self._linewidth, self._color, 1.0)

                pos += dashlength + self._gaplength

        elif self._linestyle == 'dot':
            pos = 0
            vec = self._end - self._start
            length = np.linalg.norm(vec)

            while pos < length:
                start = self._start + vec / length * pos
                self.graphic += self.sphere(start, self._linewidth, self._color, 1.0)
                pos += self._gaplength 

        # draw arrows
        vec = self._end - self._start
        vec = vec / np.linalg.norm(vec) * 0.5
        arrow_start =  self.cone(self._start, self.start - vec,0.15, self._color, self._color)
        arrow_end = self.cone(self._end, self._end + vec, 0.15, self._color, self._color)
        if self._arrow == 'end':
            self.graphic += arrow_end
        if self._arrow == 'start':
            self.graphic += arrow_start
        if self._arrow == 'both':
            self.graphic += arrow_start + arrow_end              

        return super().set_graphic()

class Helix(Point):
    def __init__(self, radius=1.0, pitch=2.0, turns=5, smoothness=100, 
                 origin=(0,0,0), direction=(0,0,1), linewidth=0.05, 
                 color=(0.93, 0.93, 0.93), name='helix'):
        """
        Draw a Helix structure

        Parameters
        ----------
        radius: float, default : 1.0
            The wide of the helix.
        pitch: float, default : 2.0
            How far the helix rises per full turn.
        turns: int, default: 5
            Number of turns. `turns` controls the length of the helix.
        smoothness: int, default : 100
            Control the smoothness.
        linewidth: float, default : 0.02
            Width of the helix.
        color: str | tuple, default : (0.93, 0.93, 0.93)
            Color of the helix, must be either HEX code or RGB color. 
        name: str, default : 'helix'
            Name of the helix used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".

        """
        super().__init__(color, name)

        self._radius = float(radius)
        self._pitch = float(pitch)
        self._turns = int(turns)
        self._smoothness = int(smoothness)
        self._origin = tuple(origin)
        self._direction = np.array(direction) / np.linalg.norm(direction)
        self._linewidth = float(linewidth)

        if len(self._origin) != 3:
            raise ValueError("'origin' must have length of 3.")
        if len(self._direction) != 3:
            raise ValueError("'direction' must have length of 3.")
        
        self.set_graphic()
    
    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius:float):
        self._radius = float(radius)
        self.set_graphic()
    
    @property
    def pitch(self):
        return self._pitch
    
    @pitch.setter
    def pitch(self, pitch:float):
        self._pitch = float(pitch)
        self.set_graphic()
    
    @property
    def turns(self):
        return self._turns

    @turns.setter
    def turns(self, turns:int):
        self._turns = int(turns)
        self.set_graphic()

    @property
    def smoothness(self):
        return self._smoothness

    @smoothness.setter
    def smoothness(self, smoothness:int):
        self._smoothness = int(smoothness)
        self.set_graphic()
    
    @property
    def origin(self):
        return self._origin
    
    @origin.setter
    def origin(self, pos:tuple):
        self._origin = tuple(pos)
        if len(self._origin) == 3:
            return self.set_graphic()
        raise ValueError("'origin' must have length of 3.")

    @property
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, direction:tuple):
        self._direction = np.array(direction)
        if len(self._direction) == 3:
            return self.set_graphic()
        raise ValueError("'direction' must have length of 3.")

    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, linewidth:float):
        self._linewidth = float(linewidth)
        self.set_graphic()
    
    def set_graphic(self):
    
        self.graphic = []

        total_segments = self._turns * self._smoothness
        angle_step = (2.0 * np.pi) / self._smoothness

        # Generate the helix coordinates and cylinders
        for i in range(total_segments):
            angle1 = i * angle_step
            angle2 = (i + 1) * angle_step

            # Point 1
            p1 = np.array([
                self._radius * np.cos(angle1),
                self._radius * np.sin(angle1),
                (self._pitch / (2 * np.pi)) * angle1
            ])

            # Point 2
            p2 = np.array([
                self._radius * np.cos(angle2),
                self._radius * np.sin(angle2),
                (self._pitch / (2 * np.pi)) * angle2
            ])
            
            # Rotate and translate
            R = _rotation(np.array([0,0,1]), self._direction)
            p1 = R @ p1 + self._origin
            p2 = R @ p2 + self._origin

            # Draw a cylinder between two points
            self.graphic += self.line(p1, p2, self._linewidth, self._color, self._color)

            # Draw two points
            self.graphic += self.sphere(p1, self._linewidth, self._color, 1.0)
            self.graphic += self.sphere(p2, self._linewidth, self._color, 1.0)
            
        return super().set_graphic()
    
class Rectangle(Patch):
    def __init__(self, center=[0,0,0], normal=[0,0,1], height=5.0, width=5.0, vertices=None, 
                 facecolor=(0.93,0.93,0.93), edgecolor=(1,0,0), edgewidth=0.02, alpha=0.5,
                 scale=1.0, name='rectangle'):

        """ 
        Draw a Rectangle shape 
        
        Parameters
        ----------
        center: list[float, float, float], default : [0,0,0]
            The Cartersian coordinates of the center point of the rectangle.
        normal: list[float, float, float], default : [0,0,1]
            The normal vector of the rectangle.
        width, height: float, default : 5.0
            Width and Height of the rectangle.
        vertices: None | list[float, float, float, float], default : None
            Four corners of the rectangle. By default, the rectangle will be defined
            by `center`, `normal`, `height`, and `width`. If `vertices` are given, 
            `center`, `normal`, `height`, and `width` will be ignored.
        facecolor: str | tuple, default : (0.93, 0.93, 0.93)
            Facecolor of the rectangle, must be either HEX code or RGB color.
        edgecolor: str | tuple, default : (1.0, 0.0, 0.0)
            Edgecolor of the rectangle, must be either HEX code or RGB color.
        edgewidth: float, default : 0.02
            Width of the edge of the rectangle.
        alpha: float, default : 0.5
            Transparency of the rectangle. `alpha` must be between 0 and 1
            (with 0 meaning the object is invisible, and 1 is opaque).
        scale: float, default : 1.0
            Scaling factor to control the size of the rectangle.
        name: str, default : 'rectangle'
            Name of the rectangle used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".
        
        """
        super().__init__(facecolor, edgecolor, edgewidth, normal, vertices, alpha, name)
        
        self._center = np.array(center)
        self._height = float(height)
        self._width = float(width)
        self._scale = float(scale)
        self._compute_vertices() if not isinstance(vertices, Iterable) else self.vertices(vertices)

        if len(self._center) != 3:
            raise ValueError("'center' must have length of 3.")

        self.set_graphic()
    
    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center:tuple):
        self._center = np.asarray(center)
        if len(self._center) == 3:
            return self._compute_vertices()
        raise ValueError("'center' must have length of 3.")
    
    @property
    def vertices(self):
        return self.p1, self.p2, self.p3, self.p4

    @vertices.setter
    def vertices(self, vertices:Iterable):
        # extract coordinates for 4 corners
        self.p1, self.p2, self.p3, self.p4 = np.array(vertices)
        # recompute center, normal, width, height, and scale
        self._center = np.average([self.p1, self.p2, self.p2, self.p3], axis=0)
        self._normal = np.cross(self.p1 - self.p3, self.p2 - self.p4)
        self._normal /= np.linalg.norm(self._normal)
        self._height = np.linalg.norm(self.p1 - self.p2)
        self._width = np.linalg.norm(self.p2 - self.p3)
        self._scale = 1.0
        self.set_graphic()
    
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, width:float):
        self._width = float(width)
        self._compute_vertices()
    
    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height:float):
        self._height = float(height)
        self._compute_vertices()
    
    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale:float):
        self._scale = float(scale)
        self._compute_vertices()
    
    def _compute_vertices(self):
        """ Process vertices from normal vector, center, height, and width """

        # Create two vectors perpendicular to normal
        v = np.array([1, 0, 0])
        if np.allclose(np.cross(v, self._normal), [0,0,0]):
            v = np.array([0, 1, 0])
        # v = np.array([1, 0, 0]) if not np.allclose(self._normal, [1, 0, 0]) else np.array([0, 1, 0])
        v_width = np.cross(self._normal, v)
        v_height = np.cross(self._normal, v_width)
        v_width /= np.linalg.norm(v_width)
        v_height /= np.linalg.norm(v_height)

        # Scale
        v_width *= self._scale
        v_height *= self._scale

        # Define 4 corners of the rectangle
        self.p1 = self._center + v_width * self._width / 2 + v_height * self._height / 2 # top-right
        self.p2 = self._center + v_width * self._width / 2 - v_height * self._height / 2 # bottom-right
        self.p3 = self._center - v_width * self._width / 2 - v_height * self._height / 2 # bottom-left
        self.p4 = self._center - v_width * self._width / 2 + v_height * self._height / 2 # top-left
        
        self.set_graphic()
    
    def set_graphic(self):

        self.graphic = []
        
        # Edges: 4 Cylinders and 4 Spheres
        if self._edgecolor is not None:
            self.graphic += self.line(self.p1, self.p2, self._edgewidth, self._edgecolor, self._edgecolor) 
            self.graphic += self.line(self.p2, self.p3, self._edgewidth, self._edgecolor, self._edgecolor) 
            self.graphic += self.line(self.p3, self.p4, self._edgewidth, self._edgecolor, self._edgecolor) 
            self.graphic += self.line(self.p4, self.p1, self._edgewidth, self._edgecolor, self._edgecolor) 

            self.graphic += self.sphere(self.p1, self._edgewidth, self._edgecolor, 1.0)
            self.graphic += self.sphere(self.p2, self._edgewidth, self._edgecolor, 1.0)
            self.graphic += self.sphere(self.p3, self._edgewidth, self._edgecolor, 1.0)
            self.graphic += self.sphere(self.p4, self._edgewidth, self._edgecolor, 1.0)

        # Face: 2 Triangles
        self.graphic += self.triangle(self.p1, self.p2, self.p3, self._normal, self._facecolor, self._alpha)
        self.graphic += self.triangle(self.p3, self.p4, self.p1, self._normal, self._facecolor, self._alpha)

        return super().set_graphic()

class Triangle(Patch):
    def __init__(self, p1=[0.577, 0.0, 0.0], p2=[-0.2887, 0.5, 0.0], p3=[-0.2887, -0.5, 0.0], 
                 facecolor=(0.93,0.93,0.93), edgecolor=(1,0,0), edgewidth=0.02, alpha=0.5, 
                 name='triangle'):

        """ 
        Draw a Triangle shape 
        
        Parameters
        ----------
        p1, p2, p3: tuple, default : [0.577, 0.0, 0.0],[-0.2887, 0.5, 0.0],[-0.2887, -0.5, 0.0]
            Three vertices of the triangle. 
        facecolor: str | tuple, default : (0.93, 0.93, 0.93)
            Facecolor of the triangle, must be either HEX code or RGB color.
        edgecolor: str | tuple, default : (1.0, 0.0, 0.0)
            Edgecolor of the triangle, must be either HEX code or RGB color.
        edgewidth: float, default : 0.02
            Width of the edge of the triangle.
        alpha: float, default : 0.5
            Transparency of the rectangle. `alpha` must be between 0 and 1
            (with 0 meaning the object is invisible, and 1 is opaque).
        name: str, default : 'triangle'
            Name of the triangle used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".
        
        """
        super().__init__(facecolor, edgecolor, edgewidth, alpha=alpha, name=name)

        self._p1, self._p2, self._p3 = tuple(p1), tuple(p2), tuple(p3)

        self.set_graphic()

    @property
    def vertices(self):
        return self.p1, self.p2, self.p3

    @vertices.setter
    def vertices(self, vertices:Iterable):
        self.p1, self.p2, self.p3 = vertices
        self.set_graphic()
    
    @property
    def p1(self):
        return self.p1

    @p1.setter
    def p1(self, pos:tuple):
        self.p1 = pos
        self.set_graphic()
    
    @property
    def p2(self):
        return self.p2

    @p2.setter
    def p2(self, pos:tuple):
        self.p2 = pos
        self.set_graphic()
    
    @property
    def p3(self):
        return self.p3

    @p3.setter
    def p3(self, pos:tuple):
        self.p3 = pos
        self.set_graphic()
    
    def _compute_vertices(self):
        pass
    
    def set_graphic(self):

        self.graphic = []

        # Edges: Cylinders and Spheres
        if self._edgecolor is not None:
            self.graphic += self.line(self._p1, self._p2, self._edgewidth, self._edgecolor, self._edgecolor) 
            self.graphic += self.line(self._p2, self._p3, self._edgewidth, self._edgecolor, self._edgecolor) 
            self.graphic += self.line(self._p3, self._p1, self._edgewidth, self._edgecolor, self._edgecolor) 

            self.graphic += self.sphere(self._p1, self._edgewidth, self._edgecolor, 1.0)
            self.graphic += self.sphere(self._p2, self._edgewidth, self._edgecolor, 1.0)
            self.graphic += self.sphere(self._p3, self._edgewidth, self._edgecolor, 1.0)

        # Face:
        self.graphic += self.triangle(self._p1, self._p2, self._p3, color=self._facecolor, alpha=self._alpha)

        return super().set_graphic()

class Circle(Patch):
    def __init__(self, center=[0,0,0], normal=[0,0,1], radius=1, edgewidth=0.02, alpha=0.5,
                 facecolor=(0.93,0.93,0.93), edgecolor=(1,0,0), name='circle'):

        """ 
        Draw a Circle shape 
        
        Parameters
        ----------
        center: list[float, float, float], default : [0,0,0]
            The Cartersian coordinates of the center point of the circle.
        normal: list[float, float, float], default : [0,0,1]
            The normal vector of the circle.
        radius: float, default : 0.1
            The radius of the circle.
        edgewidth: float, default : 0.02
            Width of the edge of the circle.
        alpha: float, default : 0.5
            Transparency of the circle. `alpha` must be between 0 and 1 
            (with 0 meaning the object is invisible, and 1 is opaque).
        facecolor: str | tuple, default : (0.93, 0.93, 0.93)
            Facecolor of the circle, must be either HEX code or RGB color.
        edgecolor: str | tuple, default : (1.0, 0.0, 0.0)
            Edgecolor of the circle, must be either HEX code or RGB color.
        name: str, default : 'circle'
            Name of the circle used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".
        
        """
        super().__init__(facecolor, edgecolor, edgewidth, normal, alpha=alpha, name=name)

        self._center = np.array(center)
        self._radius = float(radius)

        if len(self._center) != 3:
            raise ValueError("'center' must have length of 3.")
        
        self.set_graphic()
    
    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center:tuple):
        self._center = np.asarray(center)
        if len(self._center) == 3:
            return self.set_graphic()
        raise ValueError("'center' must have length of 3.")
    
    @property
    def normal(self):
        return self._normal
    
    @normal.setter
    def normal(self, normal:tuple):
        self._normal = np.array(normal)
        self._normal = self._normal / np.linalg.norm(self._normal)
        if len(self._normal) == 3:
            return self.set_graphic()
        raise ValueError("'normal' must have length of 3.")

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius:float):
        self._radius = float(radius)
        self.set_graphic()

    def set_graphic(self):

        vertices = []
        self.graphic = []
        R = _rotation(np.array([0,0,1]), self._normal)

        for angle in np.linspace(0, 2*np.pi, 360):
            dx = self._radius * np.cos(angle)
            dy = self._radius * np.sin(angle)
            rotated = R @ np.array([dx, dy, 0])
            vertices.append(rotated + self._center)

        i = 0
        while i < len(vertices)-1:
            p1 = vertices[i]
            p2 = vertices[i + 1]
            if self._edgecolor is not None:
                self.graphic += self.line(p1, p2, self._edgewidth, self._edgecolor, self._edgecolor)
            self.graphic += self.triangle(self._center, p1, p2, self._normal, self._facecolor, self._alpha)
            i += 1

        return super().set_graphic()

class Ellipse(Patch):
    def __init__(self, center=[0,0,0], normal=[0,0,1], width=2.0, height=1.0, angle=0.0, scale=1.0,
                 edgewidth=0.02, alpha=0.5, facecolor=(0.93,0.93,0.93), edgecolor=(1,0,0), 
                 name='ellipse'):
        """
        Draw an Ellipse shape

        Parameters
        ----------
        center: list[float, float, float], default : [0,0,0]
            The Cartersian coordinates of the center point of the ellipse.
        normal: list[float, float, float], default : [0,0,1]
            The normal vector of the ellipse.
        width, height: float, default : 2.0, 1.0
            The size of the ellipse.
        angle: float, default : 0.0
            Rotate the ellipse anti-clockwise, in degrees.
        scale: float, default : 1.0
            Scaling factor to control the size of the ellipse.
        edgewidth: float, default : 0.02
            Width of the edge of the ellipse.
        alpha: float, default : 0.5
            Transparency of the ellipse. `alpha` must be between 0 and 1 
            (with 0 meaning the object is invisible, and 1 is opaque).
        facecolor: str | tuple, default : (0.93, 0.93, 0.93)
            Facecolor of the ellipse, must be either HEX code or RGB color.
        edgecolor: str | tuple, default : (1.0, 0.0, 0.0)
            Edgecolor of the ellipse, must be either HEX code or RGB color.
        name: str, default : 'ellipse'
            Name of the ellipse used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".
        
        """
        super().__init__(facecolor, edgecolor, edgewidth, normal, alpha=alpha, name=name)
    
        self._center = np.array(center)
        self._width = float(width)
        self._height = float(height)
        self._angle = float(angle)
        self._scale = float(scale)

        if len(self._center) != 3:
            raise ValueError("'center' must have length of 3.")
        
        self.set_graphic()
    
    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center:tuple):
        self._center = np.asarray(center)
        if len(self._center) == 3:
            return self.set_graphic()
        raise ValueError("'center' must have length of 3.")

    @property
    def normal(self):
        return self._normal
    
    @normal.setter
    def normal(self, normal:tuple):
        self._normal = np.array(normal)
        self._normal = self._normal / np.linalg.norm(self._normal)
        if len(self._normal) == 3:
            return self.set_graphic()
        raise ValueError("'normal' must have length of 3.")
    
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, width:float):
        self._width = float(width)
        self.set_graphic()
    
    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height:float):
        self._height = float(height)
        self.set_graphic()
    
    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale:float):
        self._scale = float(scale)
        self.set_graphic()
    
    @property
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self, angle:float):
        self._angle = float(angle)
        self.set_graphic()
    
    def set_graphic(self):

        vertices = []
        self.graphic = []
        R = _rotation(np.array([0,0,1]), self._normal)

        for angle in np.linspace(0, 2*np.pi, 360):
            dx = self._scale * np.cos(angle + self._angle) * self._width
            dy = self._scale * np.sin(angle + self._angle) * self._height
            rotated = R @ np.array([dx, dy, 0])
            vertices.append(rotated + self._center)

        i = 0
        while i < len(vertices)-1:
            p1 = vertices[i]
            p2 = vertices[i + 1]
            if self._edgecolor is not None:
                self.graphic += self.line(p1, p2, self._edgewidth, self._edgecolor, self._edgecolor)
            self.graphic += self.triangle(self._center, p1, p2, self._normal, self._facecolor, self._alpha)
            i += 1

        return super().set_graphic()

class Cuboid(Shape3D):
    def __init__(self, center=(0,0,0), direction=(0,0,1),
                 width=5.0, height=5.0, depth=5.0, scale=1.0,
                 facecolor=(0.93,0.93,0.93), edgecolor=(1,0,0), edgewidth=0.02,
                 alpha=0.5, name='cuboid'):
        """ 
        Draw a Cuboid shape 
        
        Parameters
        ----------
        center: list[float, float, float], default : [0,0,0]
            The Cartersian coordinates of the center point of the cuboid.
        direction: tuple, default : (0,0,1)
            The cuboid will be aligned along `direction`.
        width, height, depth: float, default : 5.0
            The size of the cuboid.
        scale: float, default : 1.0
            Scaling factor to control the size of the cuboid.
        edgewidth: float, default : 0.02
            Width of the edge of the cuboid.
        alpha: float, default : 0.5
            Transparency of the cuboid. `alpha` must be between 0 and 1 
            (with 0 meaning the object is invisible, and 1 is opaque).
            Native render (OpenGL) in PyMol often doesn't control transparency as expected. 
            Try ray tracing instead to handle transparency more accurately.  
        facecolor: str | tuple, default : (0.93, 0.93, 0.93)
            Facecolor of the cuboid, must be either HEX code or RGB color.
        edgecolor: str | tuple, default : (1.0, 0.0, 0.0)
            Edgecolor of the cuboid, must be either HEX code or RGB color.
        name: str, default : 'cuboid'
            Name of the cuboid used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".
        
        """
        super().__init__(center, direction, width, height, depth, scale, facecolor, edgecolor, edgewidth, alpha, name)

        self._compute_vertices()
    
    def _compute_vertices(self):

        cx, cy, cz = self._center
        hx = self._scale * self._width / 2.0
        hy = self._scale * self._height / 2.0
        hz = self._scale * self._depth / 2.0

        R = _rotation(np.array([0,0,1]), self._direction)

        self._vertices = [
            R @ (cx - hx, cy - hy, cz - hz),  # 0
            R @ (cx + hx, cy - hy, cz - hz),  # 1
            R @ (cx + hx, cy + hy, cz - hz),  # 2
            R @ (cx - hx, cy + hy, cz - hz),  # 3
            R @ (cx - hx, cy - hy, cz + hz),  # 4
            R @ (cx + hx, cy - hy, cz + hz),  # 5
            R @ (cx + hx, cy + hy, cz + hz),  # 6
            R @ (cx - hx, cy + hy, cz + hz),  # 7
        ]

        self.set_graphic()
    
    def set_graphic(self):

        self.graphic = []

        # --- Edges
        if self._edgecolor is not None:
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
            ]
            
            for i1, i2 in edges:
                p1 = self._vertices[i1]
                p2 = self._vertices[i2]
                self.graphic += self.line(p1, p2, self._edgewidth, self._edgecolor, self._edgecolor)
                self.graphic += self.sphere(p1, self._edgewidth, self._edgecolor, 1.0)
                self.graphic += self.sphere(p2, self._edgewidth, self._edgecolor, 1.0)

        # --- Faces
        faces = [
            (0, 1, 2), (0, 2, 3),  # bottom (z0)
            (4, 6, 5), (4, 7, 6),  # top (z1)
            (0, 4, 5), (0, 5, 1),  # front (y0)
            (3, 2, 6), (3, 6, 7),  # back (y1)
            (0, 3, 7), (0, 7, 4),  # left (x0)
            (1, 5, 6), (1, 6, 2),  # right (x1)
        ]
        
        for face in faces:
            normal = -_compute_normal(
                self._vertices[face[0]], 
                self._vertices[face[1]], 
                self._vertices[face[2]],
            )
            self.graphic += self.triangle(
                self._vertices[face[0]],
                self._vertices[face[1]],
                self._vertices[face[2]],
                normal, self._facecolor, self._alpha
            )
        
        return super().set_graphic()

class Tetrahedron(Shape3D):
    def __init__(self, center=(0,0,0), direction=(0,0,1),
                 width=5.0, height=5.0, depth=5.0, scale=1.0,
                 facecolor=(0.93,0.93,0.93), edgecolor=(1,0,0), edgewidth=0.02,
                 alpha=0.5, name='tetrahedron'):
        """ 
        Draw a Tetrahedron shape 
        
        Parameters
        ----------
        center: list[float, float, float], default : [0,0,0]
            The Cartersian coordinates of the center point of the tetrahedron.
        direction: tuple, default : (0,0,1)
            The tetrahedron will be aligned along `direction`.
        width, height, depth: float, default : 5.0
            The size of the tetrahedron.
        scale: float, default : 1.0
            Scaling factor to control the size of the tetrahedron.
        edgewidth: float, default : 0.02
            Width of the edge of the tetrahedron.
        alpha: float, default : 0.5
            Transparency of the tetrahedron. `alpha` must be between 0 and 1 
            (with 0 meaning the object is invisible, and 1 is opaque).
            Native render (OpenGL) in PyMol often doesn't control transparency as expected. 
            Try ray tracing instead to handle transparency more accurately.  
        facecolor: str | tuple, default : (0.93, 0.93, 0.93)
            Facecolor of the tetrahedron, must be either HEX code or RGB color.
        edgecolor: str | tuple, default : (1.0, 0.0, 0.0)
            Edgecolor of the tetrahedron, must be either HEX code or RGB color.
        name: str, default : 'tetrahedron'
            Name of the tetrahedron used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".
        
        """
        super().__init__(center, direction, width, height, depth, scale, facecolor, edgecolor, edgewidth, alpha, name)

        self._compute_vertices()

    def _compute_vertices(self):
        cx, cy, cz = self._center
        hx = self._scale * self._width
        hy = self._scale * self._height
        hz = self._scale * self._depth

        self._vertices = [
            [1,  1,  1],
            [-1, -1,  1],
            [-1,  1, -1],
            [1, -1, -1],
        ]

        R = _rotation(np.array([0,0,1]), self._direction)
        for idx, vertice in enumerate(self._vertices):
            norm = np.linalg.norm(vertice)
            scaled = [
                vertice[0] / norm * (hx / 2.0),
                vertice[1] / norm * (hy / 2.0),
                vertice[2] / norm * (hz / 2.0),
            ]
            translated = [scaled[0] + cx, scaled[1] + cy, scaled[2] + cz]
            self._vertices[idx] = R @ translated

        self.set_graphic()
    
    def set_graphic(self):

        self.graphic = []

        # --- Edges
        if self._edgecolor is not None:
            edges = [
                (0, 1), (1, 2), (2, 0),
                (0, 3), (1, 3), (2, 3),
            ]

            for i1, i2 in edges:
                p1 = self._vertices[i1]
                p2 = self._vertices[i2]
                self.graphic += self.line(p1, p2, self._edgewidth, self._edgecolor, self._edgecolor)
                self.graphic += self.sphere(p1, self._edgewidth, self._edgecolor, 1.0)
                self.graphic += self.sphere(p2, self._edgewidth, self._edgecolor, 1.0)

        # --- Faces
        faces = [
            (0, 1, 2),
            (0, 3, 1),
            (0, 2, 3),
            (1, 3, 2),
        ]
        
        for face in faces:
            normal = -_compute_normal(
                self._vertices[face[0]], 
                self._vertices[face[1]], 
                self._vertices[face[2]],
            )
            self.graphic += self.triangle(
                self._vertices[face[0]],
                self._vertices[face[1]],
                self._vertices[face[2]],
                normal, self._facecolor, self._alpha
            )
        
        return super().set_graphic()

class Octahedron(Shape3D):
    def __init__(self, center=(0,0,0), direction=(0,0,1),
                 width=5.0, height=5.0, depth=5.0, scale=1.0,
                 facecolor=(0.93,0.93,0.93), edgecolor=(1,0,0), edgewidth=0.02,
                 alpha=0.5, name='octahedron'):
        """ 
        Draw an Octahedron shape 
        
        Parameters
        ----------
        center: list[float, float, float], default : [0,0,0]
            The Cartersian coordinates of the center point of the octahedron.
        direction: tuple, default : (0,0,1)
            The octahedron will be aligned along `direction`.
        width, height, depth: float, default : 5.0
            The size of the octahedron.
        scale: float, default : 1.0
            Scaling factor to control the size of the octahedron.
        edgewidth: float, default : 0.02
            Width of the edge of the octahedron.
        alpha: float, default : 0.5
            Transparency of the octahedron. `alpha` must be between 0 and 1 
            (with 0 meaning the object is invisible, and 1 is opaque).
            Native render (OpenGL) in PyMol often doesn't control transparency as expected. 
            Try ray tracing instead to handle transparency more accurately.  
        facecolor: str | tuple, default : (0.93, 0.93, 0.93)
            Facecolor of the octahedron, must be either HEX code or RGB color.
        edgecolor: str | tuple, default : (1.0, 0.0, 0.0)
            Edgecolor of the octahedron, must be either HEX code or RGB color.
        name: str, default : 'octahedron'
            Name of the octahedron used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".
        
        """
        super().__init__(center, direction, width, height, depth, scale, facecolor, edgecolor, edgewidth, alpha, name)

        self._compute_vertices()

    def _compute_vertices(self):

        cx, cy, cz = self._center
        hx = self._scale * self._width / 2.0
        hy = self._scale * self._height / 2.0
        hz = self._scale * self._depth / 2.0

        R = _rotation(np.array([0,0,1]), self._direction)

        self._vertices = [
            R @ (cx,      cy,      cz + hz),  # 0 - top
            R @ (cx,      cy,      cz - hz),  # 1 - bottom
            R @ (cx + hx, cy,      cz     ),  # 2 - +x
            R @ (cx - hx, cy,      cz     ),  # 3 - -x
            R @ (cx,      cy + hy, cz     ),  # 4 - +y
            R @ (cx,      cy - hy, cz     ),  # 5 - -y
        ]

        self.set_graphic()
            
    def set_graphic(self):

        self.graphic = []

        # --- Edges
        if self._edgecolor is not None:
            edges = [
                (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 2), (1, 3), (1, 4), (1, 5),
                (2, 4), (4, 3), (3, 5), (5, 2),
            ]
            for i1, i2 in edges:
                p1 = self._vertices[i1]
                p2 = self._vertices[i2]
                self.graphic += self.line(p1, p2, self._edgewidth, self._edgecolor, self._edgecolor)
                self.graphic += self.sphere(p1, self._edgewidth, self._edgecolor, 1.0)
                self.graphic += self.sphere(p2, self._edgewidth, self._edgecolor, 1.0)

        # --- Faces
        faces = [
            (0, 2, 4),
            (0, 4, 3),
            (0, 3, 5),
            (0, 5, 2),
            (1, 4, 2),
            (1, 3, 4),
            (1, 5, 3),
            (1, 2, 5),         
        ]
        
        for face in faces:
            normal = _compute_normal(
                self._vertices[face[0]], 
                self._vertices[face[1]], 
                self._vertices[face[2]],
            )
            self.graphic += self.triangle(
                self._vertices[face[0]],
                self._vertices[face[1]],
                self._vertices[face[2]],
                normal, self._facecolor, self._alpha
            )

        return super().set_graphic()

class Pyramid(Shape3D):
    def __init__(self, center=(0,0,0), direction=(0,0,1),
                 width=5.0, height=5.0, depth=5.0, scale=1.0,
                 facecolor=(0.93,0.93,0.93), edgecolor=(1,0,0), edgewidth=0.02,
                 alpha=0.5, name='pyramid'):
        """ 
        Draw a Pyramid shape 
        
        Parameters
        ----------
        center: list[float, float, float], default : [0,0,0]
            The Cartersian coordinates of the center point of the pyramid.
        direction: tuple, default : (0,0,1)
            The cuboid will be aligned along `direction`.
        width, height, depth: float, default : 5.0
            The size of the pyramid.
        scale: float, default : 1.0
            Scaling factor to control the size of the pyramid.
        edgewidth: float, default : 0.02
            Width of the edge of the pyramid.
        alpha: float, default : 0.5
            Transparency of the pyramid. `alpha` must be between 0 and 1 
            (with 0 meaning the object is invisible, and 1 is opaque).
            Native render (OpenGL) in PyMol often doesn't control transparency as expected. 
            Try ray tracing instead to handle transparency more accurately.  
        facecolor: str | tuple, default : (0.93, 0.93, 0.93)
            Facecolor of the pyramid, must be either HEX code or RGB color.
        edgecolor: str | tuple, default : (1.0, 0.0, 0.0)
            Edgecolor of the pyramid, must be either HEX code or RGB color.
        name: str, default : 'pyramid'
            Name of the pyramid used in PyMol. Each drawing object must have a unique name.
            `name` should not have any whitespace, e.g., 'object_1".
        
        """
        super().__init__(center, direction, width, height, depth, scale, facecolor, edgecolor, edgewidth, alpha, name)

        self._compute_vertices()

    def _compute_vertices(self):

        cx, cy, cz = self._center
        hx = self._scale * self._width / 2.0
        hy = self._scale * self._height / 2.0
        hz = self._scale * self._depth 

        R = _rotation(np.array([0,0,1]), self._direction)

        self._vertices = [
            R @ (cx - hx, cy - hy, cz     ),  # bottom-left
            R @ (cx + hx, cy - hy, cz     ),  # bottom-right
            R @ (cx + hx, cy + hy, cz     ),  # top-right
            R @ (cx - hx, cy + hy, cz     ),  # top-left
            R @ (cx     , cy     , cz + hz),  # apex point
        ]

        self.set_graphic()
    
    def set_graphic(self):

        self.graphic = []

        # --- Edges
        if self._edgecolor is not None:
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # base
                (0, 4), (1, 4), (2, 4), (3, 4),  # sides
            ]
            for i1, i2 in edges:
                p1 = self._vertices[i1]
                p2 = self._vertices[i2]
                self.graphic += self.line(p1, p2, self._edgewidth, self._edgecolor, self._edgecolor)
                self.graphic += self.sphere(p1, self._edgewidth, self._edgecolor, 1.0)
                self.graphic += self.sphere(p2, self._edgewidth, self._edgecolor, 1.0)

        # --- Faces
        faces = [
            (0, 1, 2), (0, 2, 3), # base faces
            (0, 1, 4),  # front
            (1, 2, 4),  # right
            (2, 3, 4),  # back
            (3, 0, 4),  # left
        ]
        
        for face in faces:
            normal = _compute_normal(
                self._vertices[face[0]], 
                self._vertices[face[1]], 
                self._vertices[face[2]],
            )
            self.graphic += self.triangle(
                self._vertices[face[0]],
                self._vertices[face[1]],
                self._vertices[face[2]],
                normal, self._facecolor, self._alpha
            )

        return super().set_graphic()

