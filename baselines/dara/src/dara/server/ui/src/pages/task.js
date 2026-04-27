import React from 'react';
import styled from 'styled-components';
import Layout from '../components/Layout';
import { Button, Descriptions, Divider, Modal, Spin, Tag, List, Switch, Tooltip, Typography } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import Loadable from 'react-loadable'
import parse from 'html-react-parser';
import {STATE_COLOR} from './results';

const { Text } = Typography;

const URL = process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8001/api';

const Container = styled.div`
& code {
  background-color: rgba(0, 0, 0, 0.1);
  border-radius: 6px;
  padding: .2em .4em;
  font-size: 85%;
  text-align: center;
}

.no-highlight {
  color: #555;
  background-color: rgba(0, 0, 0, 0.03);
}

.highlight {
  color: black;
}

.representative-phase {
  font-weight: bold;
  color: black;
  font-size: 1em;
  border-radius: 6px;
  border: 1px solid #555;
  padding: .2em .4em;
}

.phases-container {
  line-height: 2;
}

.error-code {
  background-color: #f9f9f9;
  padding: 10px;
  border-radius: 6px;
  border: 1px solid #ddd;
  font-color: #333;
  overflow-x: scroll;
}
`

const SpinContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  width: 100%;
`

const TitleContriner = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`

const ResultTitleContainer = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`

const Title = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`


const Plotly = Loadable({
  loader: () => import(`react-plotly.js`),
  loading: ({ timedOut }) =>
    timedOut ? (
      <blockquote>Error: Loading Plotly timed out.</blockquote>
    ) : (
      <Spin />
    ),
  timeout: 10000,
})


function RefinementPlot({ task_id, plot_id }) {
  const [plot, setPlot] = React.useState(undefined);
  const request_url = plot_id === null ? `${URL}/task/${task_id}/plot` : `${URL}/task/${task_id}/plot?idx=${plot_id}`;
  React.useEffect(() => {
    if (task_id === undefined || plot_id === undefined) {
      setPlot(undefined);
    };
    fetch(request_url)
      .then(response => {
        if (response.status === 404) {
          return null;
        }
        return response.json();
      })
      .then(response => {
        setPlot(response);
      });
  }, [task_id, plot_id]);
  return plot === undefined ? <Spin></Spin> : <Plotly {...JSON.parse(plot)} style={{ height: "100%", weight: "100%" }} />
}

function PhaseList({ phases, highlighted_phases, grouped_phases, composition_grouping }) {
  return (
    <List
      size="small"
      dataSource={phases}
      renderItem={(item, index) => (
        <List.Item key={index}>
          <List.Item.Meta
            title={`Phase ${index + 1}`}
            description={
              composition_grouping ? grouped_phases[index].map((groupedItem, idx) => (
                  <div key={idx} className='phases-container'>
                    <span className='representative-phase'>{parse(groupedItem[0])}</span> - (
                    {groupedItem[1].map((i, innerIdx) => (
                      <span key={innerIdx}>{innerIdx !== 0 ? ", " : ""}
                        <code className={i === highlighted_phases[index] ? "highlight" : "no-highlight"}>{i}</code>
                      </span>
                    ))}
                    )
                  </div>
              )) : (
                <div className='phases-container'>
                  {item.map((i, idx) => (
                    <span key={idx}>{idx !== 0 ? " | " : ""}
                      <code className={i === highlighted_phases[index] ? "highlight" : "no-highlight"}>{i}</code>
                    </span>
                  ))}
                </div>
              )
            }
          />
        </List.Item>
      )}
    />
  );
}


function TaskDetail({ data, task_id }) {
  const [plot_id, setPlotId] = React.useState(undefined);
  const [composition_grouping, setCompositionGrouping] = React.useState(true);

  return (
    <>
      <TitleContriner>
        <h3>Task details (<a onClick={() => setPlotId(null)}>Visualize final result</a>)</h3>
        {/* <Button type="primary" href={`${URL}/task/${task_id}/download`} target="_blank">Download Project File</Button> */}
      </TitleContriner>
      <Divider />
      <Descriptions bordered column={2} size='small'>
        <Descriptions.Item label="Pattern file name" span={2}>{data.task_label}</Descriptions.Item>
        <Descriptions.Item label="Best_Rwp" span={2}><Tag color="green">{data.best_rwp} %</Tag></Descriptions.Item>
        <Descriptions.Item label="Temperature" span={2}>{data.temperature || "None"}</Descriptions.Item>
        <Descriptions.Item label="Use reaction network?" span={2}>{data.use_rxn_predictor ? "Yes" : "No"}</Descriptions.Item>
        <Descriptions.Item label="Additional search options" span={2}><code style={{backgroundColor: 'transparent'}}>{JSON.stringify(data.additional_search_options)}</code></Descriptions.Item>
        <Descriptions.Item label="Compositions" span={2}>{data.precursors}</Descriptions.Item>
        <Descriptions.Item label="Final phases" span={2}>{data.final_result.phases.map((item, index) => (<span>{index ? " | " : ""}<code>{item}</code></span>))}</Descriptions.Item>
        <Descriptions.Item label="Start time" span={1}>{data.start_time}</Descriptions.Item>
        <Descriptions.Item label="End time" span={1}>{data.end_time}</Descriptions.Item>
        <Descriptions.Item label="Duration" span={2}><Tag color="cyan">{data.runtime.toFixed(2)} s</Tag></Descriptions.Item>
      </Descriptions>
      <ResultTitleContainer>
        <h3 style={{ marginTop: "16px" }}>All search results</h3>
        <div>
          Group by composition 
          <Tooltip title="If true, the phases will be grouped based on compositions, with a representative composition highlighted for each group.">
            <InfoCircleOutlined style={{marginLeft: "4px"}}/>
          </Tooltip> : 
          <Switch defaultChecked onChange={setCompositionGrouping} style={{marginLeft: "4px"}}/>
        </div>
      </ResultTitleContainer>
      <Divider />
      <Descriptions bordered column={2} size='small'>
        {
          data.all_results.map((result, index) => {
            return (<>
              <Descriptions.Item labelStyle={{ width: "25%" }} label={<>{`Search result ${index + 1} (Rwp = ${result.rwp} %)`}<br />(<a onClick={() => setPlotId(index)}>Visualize result</a>)</>} span={2}>
                {/* {result.phases.map((item, index) => (<>{index !== 0 ? <br /> : <></>}<strong>Phase {index + 1}: </strong>{item.map((i, idx) => (<span>{idx !== 0 ? " | " : ""}<code>{i}</code></span>))}</>))} */}
                <PhaseList phases={result.phases} highlighted_phases={result.highlighted_phases} grouped_phases={result.grouped_phases} composition_grouping={composition_grouping} />
              </Descriptions.Item>
            </>);
          }
          )
        }
      </Descriptions>
      <Modal title={`Refinement result ${plot_id === null ? '(final)' : plot_id + 1}`} open={plot_id !== undefined} onCancel={() => setPlotId(undefined)} footer={[]} width={"70vw"}>
        <RefinementPlot task_id={task_id} plot_id={plot_id} />
      </Modal>
    </>
  )
}

function TaskDetailedUnfinished({ data }) {
  const { status, submitted_on, start_time, end_time, error_tb } = data;
  return (<Descriptions bordered column={1} size='small'>
    <Descriptions.Item label="Name" span={1}>{data.task_label}</Descriptions.Item>
    <Descriptions.Item label="Status" span={1}><Tag color={STATE_COLOR[status]}>{status}</Tag></Descriptions.Item>
    <Descriptions.Item label="Submitted on" span={1}>{submitted_on}</Descriptions.Item>
    {start_time ? <Descriptions.Item label="Start time" span={1}>{start_time}</Descriptions.Item> : <></>}
    {end_time ? <Descriptions.Item label="End time" span={1}>{end_time}</Descriptions.Item> : <></>}
    {error_tb ? <Descriptions.Item label="Error Traceback" span={1}><div class="error-code"><pre style={{ whiteSpace: "break-spaces", fontSize: "0.9em" }}>{error_tb}</pre></div></Descriptions.Item> : <></>}
  </Descriptions>)
}


function TaskPage({ location }) {
  const queryParameters = new URLSearchParams(location.search)  
  const task_id = queryParameters.get('id')
  const [error, setError] = React.useState("");

  const [data, setData] = React.useState(undefined);
  console.log(data);
  React.useEffect(() => {
    // handle 404 error
    fetch(`${URL}/task/${task_id}`)
      .then(response => {
        if (response.status === 404 || response.status === 400) {
          return response.json().then(data => setError(data.detail)).then(() => null);
        }
        return response.json();
      })
      .then(data => setData(data));
  }, [task_id]);

  return (
    <Layout hasSider={false} title="Result">
      <Container>
        <Title>
        <h2>Task {task_id}</h2>
        <div>
          <Button type="primary" href={`https://github.com/CederGroupHub/dara/issues/new`} target="_blank">Open Github Issue</Button>
        </div>
        </Title>
        {
          data === null ? <h3>{error}</h3> : <></>
        }
        {
          data === undefined ? <SpinContainer><Spin size='large'></Spin></SpinContainer> : <></> 
        }
        {
          data && data.status === "COMPLETED" ? <TaskDetail data={data} task_id={task_id} /> : <></> 
        }
        {
          data && data.status !== "COMPLETED" ? <TaskDetailedUnfinished data={data} /> : <></>
        }
      </Container>
    </Layout>
  )
}

export default TaskPage;
