import absl.app
import os
from domrand.define_flags import FLAGS
from domrand.sim_manager import SimManager
from domrand.utils.data import write_data_with_sim


def main(_):
    xml_path = FLAGS.xml
    
    try:
        # Create SimManager with proper rendering
        sim_manager = SimManager(
            filepath=xml_path,
            gpu_render=FLAGS.gpu_render,
            gui=FLAGS.gui,
            display_data=FLAGS.display_data
        )
        
        print(f"SimManager initialized from {xml_path}")
        
        # Write data using the SimManager
        if not FLAGS.gui:
            write_data_with_sim(sim_manager, FLAGS.data_path)

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    absl.app.run(main)





