<?xml version='1.0' encoding='utf-8'?>
<scheme version="2.0" title="" description="">
	<nodes>
		<node id="0" name="Datasets" qualified_name="Orange.widgets.data.owdatasets.OWDataSets" project_name="Orange3" version="" title="Datasets" position="(150, 150)" />
		<node id="1" name="Data Table" qualified_name="Orange.widgets.data.owtable.OWTable" project_name="Orange3" version="" title="Data Table" position="(360.0, 72.0)" />
		<node id="2" name="Linear Regression" qualified_name="Orange.widgets.model.owlinearregression.OWLinearRegression" project_name="Orange3" version="" title="Linear Regression" position="(317.0, 233.0)" />
		<node id="3" name="Predictions" qualified_name="Orange.widgets.evaluate.owpredictions.OWPredictions" project_name="Orange3" version="" title="Predictions" position="(554.0, 209.0)" />
		<node id="4" name="Random Forest" qualified_name="Orange.widgets.model.owrandomforest.OWRandomForest" project_name="Orange3" version="" title="Random Forest" position="(311.0, 347.0)" />
		<node id="5" name="Neural Network" qualified_name="Orange.widgets.model.owneuralnetwork.OWNNLearner" project_name="Orange3" version="" title="Neural Network" position="(308.0, 455.0)" />
	</nodes>
	<links>
		<link id="0" source_node_id="0" sink_node_id="1" source_channel="Data" sink_channel="Data" enabled="true" source_channel_id="data" sink_channel_id="data" />
		<link id="1" source_node_id="0" sink_node_id="2" source_channel="Data" sink_channel="Data" enabled="true" source_channel_id="data" sink_channel_id="data" />
		<link id="2" source_node_id="2" sink_node_id="3" source_channel="Model" sink_channel="Predictors" enabled="true" source_channel_id="model" sink_channel_id="predictors" />
		<link id="3" source_node_id="0" sink_node_id="3" source_channel="Data" sink_channel="Data" enabled="true" source_channel_id="data" sink_channel_id="data" />
		<link id="4" source_node_id="0" sink_node_id="4" source_channel="Data" sink_channel="Data" enabled="true" source_channel_id="data" sink_channel_id="data" />
		<link id="5" source_node_id="4" sink_node_id="3" source_channel="Model" sink_channel="Predictors" enabled="true" source_channel_id="model" sink_channel_id="predictors" />
		<link id="6" source_node_id="0" sink_node_id="5" source_channel="Data" sink_channel="Data" enabled="true" source_channel_id="data" sink_channel_id="data" />
		<link id="7" source_node_id="5" sink_node_id="3" source_channel="Model" sink_channel="Predictors" enabled="true" source_channel_id="model" sink_channel_id="predictors" />
	</links>
	<annotations />
	<thumbnail />
	<node_properties>
		<properties node_id="0" format="literal">{'controlAreaVisible': True, 'domain': None, 'filter_hint': 'hous', 'header_state': b'\x00\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04&gt;\x00\x00\x00\x07\x01\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00d\xff\xff\xff\xff\x00\x00\x00\x81\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x1c\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x01\x0b\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00R\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00O\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00B\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00d\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x01\xd0\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00d', 'language': 'English', 'savedWidgetGeometry': b'\x01\xd9\xd0\xcb\x00\x03\x00\x00\xff\xff\xf2\x99\x00\x00\x01\x03\xff\xff\xf6\xe6\x00\x00\x03\x16\xff\xff\xf2\x9a\x00\x00\x01"\xff\xff\xf6\xe5\x00\x00\x03\x15\x00\x00\x00\x02\x00\x00\x00\x00\x07\x80\xff\xff\xf2\x9a\x00\x00\x01"\xff\xff\xf6\xe5\x00\x00\x03\x15', 'selected_id': 'california_housing.xlsx', 'splitter_state': b'\x00\x00\x00\xff\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x01,\x00\x00\x00\xc8\x01\xff\xff\xff\xff\x01\x00\x00\x00\x02\x00', '__version__': 2}</properties>
		<properties node_id="1" format="literal">{'auto_commit': True, 'color_by_class': True, 'controlAreaVisible': True, 'savedWidgetGeometry': b'\x01\xd9\xd0\xcb\x00\x03\x00\x00\xff\xff\xf1}\x00\x00\x01|\xff\xff\xf7`\x00\x00\x03\x8f\xff\xff\xf1~\x00\x00\x01\x9b\xff\xff\xf7_\x00\x00\x03\x8e\x00\x00\x00\x02\x00\x00\x00\x00\x07\x80\xff\xff\xf1~\x00\x00\x01\x9b\xff\xff\xf7_\x00\x00\x03\x8e', 'select_rows': True, 'show_attribute_labels': True, 'show_distributions': False, 'stored_selection': {'rows': [], 'columns': []}, 'stored_sort': [], '__version__': 1}</properties>
		<properties node_id="2" format="literal">{'alpha_index': 0, 'auto_apply': True, 'autosend': True, 'controlAreaVisible': True, 'fit_intercept': True, 'l2_ratio': 0.5, 'learner_name': '', 'reg_type': 0, 'ridge': False, 'savedWidgetGeometry': b'\x01\xd9\xd0\xcb\x00\x03\x00\x00\xff\xff\xf4\x13\x00\x00\x01F\xff\xff\xf5m\x00\x00\x02\xd3\xff\xff\xf4\x14\x00\x00\x01e\xff\xff\xf5l\x00\x00\x02\xd2\x00\x00\x00\x02\x00\x00\x00\x00\x07\x80\xff\xff\xf4\x14\x00\x00\x01e\xff\xff\xf5l\x00\x00\x02\xd2', '__version__': 1}</properties>
		<properties node_id="3" format="pickle">gASVpAIAAAAAAAB9lCiMEmNvbnRyb2xBcmVhVmlzaWJsZZSIjBNzYXZlZFdpZGdldEdlb21ldHJ5
lENCAdnQywADAAD///FFAAABC///9CIAAAOW///xRgAAASr///QhAAADlQAAAAIAAAAAB4D///FG
AAABKv//9CEAAAOVlIwJc2VsZWN0aW9ulF2UjAtzaG93X3Njb3Jlc5SIjAtzY29yZV90YWJsZZR9
lIwQc2hvd19zY29yZV9oaW50c5R9lCiMBk1vZGVsX5SIjAZUcmFpbl+UiYwFVGVzdF+UiYwCQ0GU
iIwXUHJlY2lzaW9uUmVjYWxsRlN1cHBvcnSUiIwLVGFyZ2V0U2NvcmWUiIwJUHJlY2lzaW9ulIiM
BlJlY2FsbJSIjAJGMZSIjANBVUOUiIwHTG9nTG9zc5SJjAtTcGVjaWZpY2l0eZSJjBdNYXR0aGV3
c0NvcnJDb2VmZmljaWVudJSIjANNU0WUiIwEUk1TRZSIjANNQUWUiIwETUFQRZSIjAJSMpSIjAZD
VlJNU0WUiYwPQ2x1c3RlcmluZ1Njb3JllIiMClNpbGhvdWV0dGWUiIwXQWRqdXN0ZWRNdXR1YWxJ
bmZvU2NvcmWUiHVzjAtfX3ZlcnNpb25fX5RLAowQY29udGV4dF9zZXR0aW5nc5RdlIwVb3Jhbmdl
d2lkZ2V0LnNldHRpbmdzlIwHQ29udGV4dJSTlCmBlH2UKIwGdmFsdWVzlH2UKIwXc2hvd19wcm9i
YWJpbGl0eV9lcnJvcnOUiIwPc2hvd19yZWdfZXJyb3JzlEsBjAtzaG93bl9wcm9ic5RLAIwMdGFy
Z2V0X2NsYXNzlIwWKEF2ZXJhZ2Ugb3ZlciBjbGFzc2VzKZRoB32UaCFLAnWMB2NsYXNzZXOUKXVi
YXUu
</properties>
		<properties node_id="4" format="literal">{'auto_apply': True, 'class_weight': False, 'controlAreaVisible': True, 'index_output': 0, 'learner_name': '', 'max_depth': 3, 'max_features': 5, 'min_samples_split': 5, 'n_estimators': 10, 'savedWidgetGeometry': b'\x01\xd9\xd0\xcb\x00\x03\x00\x00\xff\xff\xf3\xfd\x00\x00\x01V\xff\xff\xf5\x82\x00\x00\x02\xc3\xff\xff\xf3\xfe\x00\x00\x01u\xff\xff\xf5\x81\x00\x00\x02\xc2\x00\x00\x00\x02\x00\x00\x00\x00\x07\x80\xff\xff\xf3\xfe\x00\x00\x01u\xff\xff\xf5\x81\x00\x00\x02\xc2', 'use_max_depth': False, 'use_max_features': False, 'use_min_samples_split': True, 'use_random_state': False, '__version__': 1}</properties>
		<properties node_id="5" format="literal">{'activation_index': 2, 'alpha_index': 10, 'auto_apply': True, 'controlAreaVisible': True, 'hidden_layers_input': '10,', 'learner_name': '', 'max_iterations': 15, 'replicable': True, 'savedWidgetGeometry': b'\x01\xd9\xd0\xcb\x00\x03\x00\x00\xff\xff\xf7\x0f\x00\x00\x02\x8f\xff\xff\xf8V\x00\x00\x03\xd4\xff\xff\xf7\x10\x00\x00\x02\xae\xff\xff\xf8U\x00\x00\x03\xd3\x00\x00\x00\x02\x00\x00\x00\x00\x07\x80\xff\xff\xf7\x10\x00\x00\x02\xae\xff\xff\xf8U\x00\x00\x03\xd3', 'solver_index': 2, '__version__': 2}</properties>
	</node_properties>
	<session_state>
		<window_groups />
	</session_state>
</scheme>