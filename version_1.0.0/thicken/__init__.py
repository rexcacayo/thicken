bl_info = {'name': 'Thicken', 'author': 'Ricardo Lugaresi', 'version': (1, 0, 0),
           'blender': (4, 2, 0), 'location': 'Vista 3D > N > Thicken',
           'description': 'Engorda solo las zonas demasiado finas para imprimir, sin tocar el resto',
           'category': 'Object'}
import bpy
from . import common, overlay, thickening

NOZZLES = [('0.2', '0.2', ''), ('0.25', '0.25', ''), ('0.4', '0.4', ''), ('0.6', '0.6', ''), ('0.8', '0.8', '')]


def _stale(self, context):
    context.scene.thicken_props.checked = False
    overlay.clear()


def _wall(p):
    return p.min_wall if p.min_wall > 0 else 2 * float(p.nozzle)


class ThickenProps(bpy.types.PropertyGroup):
    model_unit: bpy.props.EnumProperty(name='Unidades', items=common.UNIT_ITEMS, default='AUTO', update=_stale)
    nozzle: bpy.props.EnumProperty(name='Boquilla', items=NOZZLES, default='0.4', update=_stale)
    min_wall: bpy.props.FloatProperty(name='Grosor mínimo (mm)', description='0 = dos pasadas de la boquilla', default=0.0, min=0.0, max=10, precision=2, update=_stale)
    checked: bpy.props.BoolProperty(default=False)
    r_red: bpy.props.FloatProperty()
    r_amber: bpy.props.FloatProperty()
    r_model: bpy.props.StringProperty()
    done: bpy.props.StringProperty()


def _model(context):
    try:
        return common.pick_model(context)
    except common.AddonError:
        return None


class OBJECT_OT_thicken_check(bpy.types.Operator):
    """Marca en el modelo las zonas más finas que el mínimo"""
    bl_idname = 'object.thicken_check'; bl_label = 'Buscar zonas finas'; bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return common.poll_object_mode(cls, context)

    def execute(self, context):
        p = context.scene.thicken_props
        try:
            obj = common.pick_model(context)
            r = thickening.analyze(context, obj, _wall(p), float(p.nozzle), p.model_unit)
        except common.AddonError as exc:
            self.report({'ERROR'}, str(exc)); return {'CANCELLED'}
        p.r_red, p.r_amber, p.r_model, p.checked, p.done = r['red_share'], r['amber_share'], obj.name, True, ''
        overlay.show(r['red_pts'], r['amber_pts'])
        return {'FINISHED'}


class OBJECT_OT_thicken_apply(bpy.types.Operator):
    """Engorda las zonas finas hasta el mínimo; el original queda oculto"""
    bl_idname = 'object.thicken_apply'; bl_label = 'Engordar zonas finas'; bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return common.poll_object_mode(cls, context)

    def execute(self, context):
        p = context.scene.thicken_props
        try:
            obj = common.pick_model(context)
            out, share, grown = thickening.thicken(context, obj, _wall(p), p.model_unit)
            r = thickening.analyze(context, out, _wall(p), float(p.nozzle), p.model_unit)
        except common.AddonError as exc:
            self.report({'ERROR'}, str(exc)); return {'CANCELLED'}
        p.r_red, p.r_amber, p.r_model, p.checked = r['red_share'], r['amber_share'], out.name, True
        overlay.show(r['red_pts'], r['amber_pts'])
        p.done = f'{share * 100:.1f} % del modelo engordado (hasta +{grown:.2f} mm)'
        self.report({'INFO'}, p.done)
        return {'FINISHED'}


class OBJECT_OT_thicken_clear(bpy.types.Operator):
    """Quita los puntos de colores"""
    bl_idname = 'object.thicken_clear'; bl_label = 'Quitar marcas'

    def execute(self, context):
        overlay.clear(); context.scene.thicken_props.checked = False
        return {'FINISHED'}


class VIEW3D_PT_thicken(bpy.types.Panel):
    bl_label = 'Thicken'; bl_idname = 'VIEW3D_PT_thicken'; bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'Thicken'

    def draw(self, context):
        layout = self.layout; p = context.scene.thicken_props
        if common.draw_mode_warning(layout, context):
            return
        model = _model(context)
        box = layout.box()
        box.label(text='1 · Modelo', icon='MESH_DATA')
        if model is None:
            r = box.row(); r.alert = True; r.label(text='Selecciona el modelo', icon='ERROR')
        else:
            box.label(text=model.name, icon='OBJECT_DATA')
        r = box.row(align=True); r.label(text='Boquilla'); r.prop(p, 'nozzle', expand=True)
        box.prop(p, 'min_wall')
        box.label(text=f'Mínimo usado: {_wall(p):.2f} mm', icon='INFO')
        col = box.column(); col.scale_y = 1.3
        col.operator('object.thicken_check', icon='VIEWZOOM')
        if p.checked and model is not None and model.name == p.r_model:
            box = layout.box()
            r = box.row(); r.alert = p.r_red > 0.005
            r.label(text=f'No sale (< {float(p.nozzle):g} mm): {p.r_red * 100:.1f} %', icon='CANCEL' if p.r_red > 0.005 else 'CHECKMARK')
            box.label(text=f'Frágil (< {_wall(p):.2f} mm): {p.r_amber * 100:.1f} %', icon='ERROR' if p.r_amber > 0.01 else 'CHECKMARK')
            box.operator('object.thicken_clear', icon='X')
        box = layout.box()
        box.label(text='2 · Engordar', icon='MOD_SOLIDIFY')
        col = box.column(); col.scale_y = 1.6
        col.operator('object.thicken_apply', icon='MOD_SOLIDIFY')
        t = box.column(align=True); t.scale_y = .8
        t.label(text='Solo cambian las zonas finas; el resto')
        t.label(text='queda igual. Después, pásalo por Mesh Doctor.')
        if p.done:
            layout.label(text=p.done, icon='CHECKMARK')


classes = (ThickenProps, OBJECT_OT_thicken_check, OBJECT_OT_thicken_apply, OBJECT_OT_thicken_clear, VIEW3D_PT_thicken)


def register():
    common.register_classes(classes, 'thicken_props', ThickenProps)
    overlay.register()


def unregister():
    overlay.unregister()
    common.unregister_classes(classes, 'thicken_props')
