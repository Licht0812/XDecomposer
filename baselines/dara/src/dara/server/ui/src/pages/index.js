import React from 'react';
import { useState } from 'react';
import styled from 'styled-components';
import Layout from '../components/Layout';
import { Button, Form, Radio, Input, Upload, Alert, message, Modal, Select, Switch, InputNumber } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

const URL = process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8001/api';

const { confirm } = Modal;
const { Dragger } = Upload;

const Container = styled.div`
  font-size: 1rem;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  align-content: center;


  & code {
    background-color: rgba(0, 0, 0, 0.1);
    border-radius: 6px;
    padding: .2em .4em;
    font-size: 85%;
    text-align: center;
  }

  .center-text {
    text-align: center;
  }

  .center {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }
`

const INSTRUMENT_NAMES = [
  'synchrotron',
  'Aeris-fds-Pixcel1d-Medipix3',
  'LBL-d8-LynxEyeXE',
  'd8-solxe-fds-0600',
  'siemens-d5000-fds1mm',
  'RMS-D8-ADS-15-LynxEyeXE',
  'RMS-D8-ADS-15-Glass-LynxEyeXE',
  'x8-apex2-fds-10',
  'xpert-xcel-ads-10mm',
  'd8-lynxeye-fds-02',
  'cubix-ads-15mm',
  'xpert-xcel-htk-ads-10',
  'siemens-d5000-fds1mm-2',
  'siemens-d5000-fds2mm',
  'rigaku-ultima',
  'd2-ssd160-fds-1',
  'cubix-ads-10mm',
  'Rigaku-Miniflex-600-DTEXultra2-fds',
  'RMS-D8-Capillary-500um-LynxEyeXE',
  'd8-fds-02-LynxEyeXE',
  'pw1800-fds',
  'RMS-D8-FDS-03-LynxEyeXE',
  'd8-lynxeye-fds-06mm',
  'xpert-xcel-htk-fds-0125',
  'PW3040-FDS-ADS-Xcelerator',
  'Rigaku-SmartLab-CBO-BB-FDS05deg',
  'd8-lynxeye-fds-05mm',
  'd8-lynxeye-ads-1mm',
  'D8_6Div_4SS',
  'Rigaku-Miniflex-gen5-Dtex-varfix',
  'Rigaku-Miniflex-gen5-Dtex-var',
  'xpert-xcel-fds-1000',
  'pw1800-ads-10mm',
  'Rigaku-Miniflex',
  'd8-solxe-vds-12mm',
  'xpert-xcel-fds-0125',
  'xpert-xcel-ads-10mm-Ge',
  'xpert-xcel-fds-0250',
  'xpert-pixcel-0500'
];

const WAVELENGTHS = [
  ["Cu", 1.540598],
  ["Co", 1.789010],
  ["Cr", 2.289760],
  ["Fe", 1.936042],
];

function Submission() {
  const [form] = Form.useForm();
  const [rxn_ntwk, setUseRxnntwk] = useState(false);
  const [synchrotron, setSynchrotron] = useState(false);
  // const [synchrotronMessage, setSynchrotronMessage] = useState("");
  const [ previousInstrumentSelected, setPreviousInstrumentSelected ] = useState("Aeris-fds-Pixcel1d-Medipix3");
  const [msg, setMsg] = useState(null);
  const [messageApi, contextHolder] = message.useMessage();
  const [spinning, setSpinning] = React.useState(false);
  const layout = {
    labelCol: { span: 10 },
    wrapperCol: { span: 14 },
  };

  const onValuesChange = (changedValues, allValues) => {
    setUseRxnntwk(allValues.use_rxn_predictor);
  }

  const onSubmit = () => {
    const formData = new FormData();
    const values = form.getFieldsValue();

    for (const key in values) {
      if (values[key] === undefined) {
        return;
      }
      if (key === 'precursor_formulas') {
        formData.append(key, JSON.stringify(
          values[key].split(',').map(item => item.trim())
        ));
      } else if (key === 'temperature') {
        formData.append(key, rxn_ntwk ? values[key] : -1000);
      } else if (key.startsWith('wavelength')) {
        if (!synchrotron) {
          formData.append('wavelength', values[key]);
        } else {
          formData.append('wavelength', Number(values[key]) / 10);
        }
      } else {
        formData.append(key, values[key]);
      }
    }

    for (const key of formData.keys()) {
      console.log(key, formData.get(key));
    }

    const abort_control = new AbortController()
    setMsg(null);
    setSpinning(true);

    confirm({
      title: 'Do you want to submit the task?',
      onOk() {
        return fetch(`${URL}/submit`, {
          method: 'POST',
          body: formData,
          signal: abort_control.signal,
        })
          .then(response => response.json())
          .then(data => {
            if (data.message === "submitted") {
              form.resetFields();
              messageApi.open({
                type: 'success',
                content: 'The refinement task has been submitted successfully!',
              });
            } else {
              setMsg(JSON.stringify(data));
            }
          })
          .finally(() => setSpinning(false))
          .catch(error => {
            if (error.name === 'AbortError') {
              return;
            }
            setMsg(error);
          });
      },
      onCancel() { abort_control.abort(); setSpinning(false); },
    });
  }

  return (
    <Layout hasSider={false} title="Submit">
      {contextHolder}
      <Container>
        {
          msg ? <Alert message={msg} type="error" showIcon closable style={{ width: '50%' }} /> : <></>
        }
        <h1>Submit refinement run to Dara system</h1>
        <Form
          {...layout}
          style={{ width: '60%' }}
          form={form}
          initialValues={{ use_rxn_predictor: false, temperature: null, "wavelength_lab": 'Cu', instrument_profile: "Aeris-fds-Pixcel1d-Medipix3", synchrotron: false }}
          onValuesChange={onValuesChange}
        >
          <Form.Item label="User" name="user" tooltip="The username for this refinement. It can be any string that will be shown with the result later." rules={[{ required: true, message: 'Please input your username!' }]}>
            <Input placeholder="Username that submits this search run" />
          </Form.Item>
          <Form.Item label="Use reaction network?" name="use_rxn_predictor" tooltip="Whether to use reaction-network to filter the phases based on energy.">
            <Radio.Group value={rxn_ntwk} buttonStyle="solid">
              <Radio.Button value={true}>Yes</Radio.Button>
              <Radio.Button value={false}>No</Radio.Button>
            </Radio.Group>
          </Form.Item>
          <Form.Item label={rxn_ntwk ? "Precursors Formulas" : "Included elements"} name="precursor_formulas" rules={[{ required: true, message: 'Please input the formula!' }]} tooltip={rxn_ntwk ? "Example: Bi2O3, Fe2O3" : "Example: Bi,Fe,O"}>
            <Input placeholder="If more than one, separate them by comma" />
          </Form.Item>
          <Form.Item label="Temperature" name="temperature" rules={[{ required: rxn_ntwk, message: 'Please input the temperature!' }]} hidden={rxn_ntwk ? false : true}>
            <Input placeholder="The temperature (°C) to synthesize the sample." />
          </Form.Item>
          <Form.Item label="Instrument Profile" name="instrument_profile" tooltip="The instrument name for collecting the pattern.">
            <Select
              showSearch
              optionFilterProp="children"
              filterOption={(input, option) =>
                (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
              }
              options={INSTRUMENT_NAMES.map(name => ({ label: name, value: name }))}
            />
          </Form.Item>
          <Form.Item label="Synchrotron" name="synchrotron" tooltip="Whether the data is collected from synchrotron.">
            <Switch onChange={(checked) => {
              setSynchrotron(checked);
              // save the previous instrument name
              if (checked) {
                setPreviousInstrumentSelected(form.getFieldValue('instrument_profile'));
                form.setFieldsValue({ instrument_profile: "synchrotron" });
              } else {
                form.setFieldsValue({ instrument_profile: previousInstrumentSelected });
              }
            }} />
            {/* <span style={{color: "red", marginLeft: "16px"}}>{synchrotronMessage}</span> */}
          </Form.Item>
          {!synchrotron ? (
            <Form.Item label="Wavelength" name="wavelength_lab" tooltip="The wavelength in nm of X-ray for collecting the pattern." rules={[{ required: true, message: 'Please input the wavelength!' }]}>
              <Select
                showSearch
                optionFilterProp="children"
                filterOption={(input, option) =>
                  (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                }
                options={WAVELENGTHS.map((name) => ({ value: name[0], label: `${name[0]} (${name[1].toFixed(4)} Å)` }))}
              />
            </Form.Item>
          ) : (
            <Form.Item label="Wavelength" name="wavelength_sync" tooltip="The wavelength of X-ray for collecting the pattern." rules={[{ required: true, message: 'Please input a valid wavelength!', pattern: /^\d+(\.\d+)?$/ }]}>
              <Input placeholder="The wavelength of X-ray for collecting the pattern." suffix="Å" />
            </Form.Item>
          )
          }
          <Form.Item label="Upload XRD pattern file" name="pattern_file" rules={[{ required: true, message: 'Please upload the XRD pattern file!' }]} tooltip="Currently support xy/xrdml/raw/txt file">
            <Dragger
              maxCount={1}
              beforeUpload={() => false}
              onChange={(f) => {
                form.setFieldValue('pattern_file', f.file);
              }}
              onDrop={(f) => {
                form.setFieldValue('pattern_file', f.file);
              }}
            >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">Click or drag file to this area to upload</p>
                <p className="ant-upload-hint">
                  Supprted file types: xy(e) / xrdml / raw / txt
                </p>
              {/* </div> */}

            </Dragger>
          </Form.Item>
          <Form.Item className='center'>
            <Button type="primary" htmlType="submit" size="large" onClick={onSubmit} disabled={spinning}>
              Submit
            </Button>
          </Form.Item>
        </Form>
      </Container>

    </Layout>
  )
}

export default Submission;
