from vyper import v


def get_config():
	v.set_config_name('secrets')
	v.add_config_path('.streamlit')
	v.read_in_config()
	return v
