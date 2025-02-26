#ifndef __UTILS_INTERSECTION_HPP__
#define __UTILS_INTERSECTION_HPP__


#include <opencv2/opencv.hpp>
#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>
#include <g3log/std2_make_unique.hpp>
#include <g3log/logmessage.hpp>

float distPoint( cv::Point p1, cv::Point p2 ) ;
float distPoint(cv::Point2f p1,cv::Point2f p2) ;
bool segementIntersection(cv::Point p0_seg0,cv::Point p1_seg0,cv::Point p0_seg1,cv::Point p1_seg1,cv::Point * intersection) ;
bool segementIntersection(cv::Point2f p0_seg0,cv::Point2f p1_seg0,cv::Point2f p0_seg1,cv::Point2f p1_seg1,cv::Point2f * intersection) ;

bool pointInPolygon(cv::Point p,const cv::Point * points,int n) ;
bool pointInPolygon(cv::Point2f p,const cv::Point2f * points,int n) ;


#define MAX_POINT_POLYGON 64
struct Polygon {
	cv::Point pt[MAX_POINT_POLYGON];
	int     n;

	Polygon(int n_ = 0 ) { assert(n_>= 0 && n_ < MAX_POINT_POLYGON); n = n_;}
	virtual ~Polygon() {}

	void clear() { n = 0; }
	void add(const cv::Point &p) {if(n < MAX_POINT_POLYGON) pt[n++] = p;}
	void push_back(const cv::Point &p) {add(p);}
	int size() const { return n;}
	cv::Point getCenter() const ;
	const cv::Point & operator[] (int index) const { assert(index >= 0 && index < n); return pt[index];}
	cv::Point& operator[] (int index) { assert(index >= 0 && index < n); return pt[index]; }
	void pointsOrdered() ;
	float area() const ;
	bool pointIsInPolygon(cv::Point p) const ;
};


void intersectPolygon( const cv::Point * poly0, int n0,const cv::Point * poly1,int n1, Polygon & inter ) ;
void intersectPolygon( const Polygon & poly0, const Polygon & poly1, Polygon & inter ) ;
void intersectPolygonSHPC(const Polygon * sub,const Polygon* clip,Polygon* res) ;
void intersectPolygonSHPC(const Polygon & sub,const Polygon& clip,Polygon& res) ;




struct AnniwoTimeLog
{
    double configInterval; //ms
    std::chrono::steady_clock::time_point lastTP;
};

struct AnniwoStay
{
    bool isMotionless; //isMotionless:是否从静止开始计算停留时间
    int staySec;  //要求停留秒数，超过该时间再报警
};

struct AnniwoSafeEreaConcernTypes
{
    std::vector<std::string> validtypes;
    std::vector<std::string> excludeTypes;
};

struct ANNIWO_JSON_CONF_CLASS
{
// std::unordered_map<int, std::vector<std::string> > id_func_cap = {
//     { 23,{"smoke","safeErea"} },
//     { 5,{"helmet","fire"} },
//     { 43,{"safeErea","fire"} } };
std::unordered_map<int, std::vector<std::string> > id_func_cap;

//interval
//camId,{func,<interval,interval in seconds and temp counter>}

std::unordered_map<int,std::unordered_map<std::string, AnniwoTimeLog >  > interval_conf_map;
//camId,{func,<isMotionless,stay in seconds>}
std::unordered_map<int,std::unordered_map<std::string, AnniwoStay>  > stay_conf_map;

//valid area.
//camId,{func,Polygon}
std::unordered_map<int,std::unordered_map<std::string, Polygon>  > validArea_conf_map;
//valid types
//camId,{func,{Vector<String>,Vector<String>}}
std::unordered_map<int,std::unordered_map<std::string, AnniwoSafeEreaConcernTypes>  >  validtypes_map;
//valid peroids_hour_map 
//valid peroids_week_map
//camId,{func,Vector<Polygon>}
//camId,{func,Vector<std::string>}
std::unordered_map<int,std::unordered_map<std::string, std::vector<Polygon>>  >  validperoids_hour_map;
std::unordered_map<int,std::unordered_map<std::string, std::vector<std::string>>  >  validperoids_week_map;


//camId,{func,stringtaskid}
std::unordered_map<int,std::unordered_map<std::string, std::string>  > taskid_conf_map;

//camId,{func,facedatasetpath}
std::unordered_map<int,std::unordered_map<std::string, std::string>  > facedatasetpath_conf_map;


//camId,{func,eventUrl}
std::unordered_map<int,std::unordered_map<std::string, std::string>  > eventUrl_conf_map;

//This is for absence function's startCondition 
//camId,{func,startCondition}
std::unordered_map<int,std::unordered_map<std::string, std::string>  > absenceStartConition_conf_map;

};


inline void getTaskId(const ANNIWO_JSON_CONF_CLASS* globalJsonConfObjPtr,int camID,const std::string strFuncname,/*out*/std::string& taskIdstr)
{
	//camId,{func,stringtaskid}
	const std::unordered_map<int,std::unordered_map<std::string, std::string>  >* taskid_conf_map_ptr= &globalJsonConfObjPtr->taskid_conf_map;

    //取得taskId设置
    //camId,{func,taskId}
    std::unordered_map<int,std::unordered_map<std::string, std::string>  >::const_iterator got_id_func_cap = taskid_conf_map_ptr->find(camID);

    if (got_id_func_cap == taskid_conf_map_ptr->end())
    {
        ANNIWOLOG(INFO) << "not found in taskid_conf_map,camID:" <<camID;
    }
    else
    {
        const std::unordered_map<std::string, std::string>& conf_map =got_id_func_cap->second;
        std::unordered_map<std::string, std::string>::const_iterator got_id_func_cap2 = conf_map.find(strFuncname);
        if (got_id_func_cap2 == conf_map.end())
        {
            ANNIWOLOG(INFO) << "not found "<< strFuncname<<" in taskid_conf_map,camID:" <<camID;
        }
        else
        {
            taskIdstr = got_id_func_cap2->second ;
        }
    }
}

//strFuncname:查找配置关键字
//suffix:添加在配置之后
//submitUrl:输出
inline void getEventUrl(const ANNIWO_JSON_CONF_CLASS* globalJsonConfObjPtr,int camID,const std::string strFuncname,const std::string suffix,/*out*/std::string& submitUrl)
{
	//camId,{func,stringenventurl}
	const std::unordered_map<int,std::unordered_map<std::string, std::string>  >* eventUrl_conf_map_ptr= &globalJsonConfObjPtr->eventUrl_conf_map;

    //取得eventUrl设置
    //camId,{func,eventUrl}
    std::unordered_map<int,std::unordered_map<std::string, std::string>  >::const_iterator got_id_func_cap = eventUrl_conf_map_ptr->find(camID);

    if (got_id_func_cap == eventUrl_conf_map_ptr->end())
    {
        ANNIWOLOG(INFO) << "not found in eventUrl_conf_map,camID:" <<camID;
    }
    else
    {
        const std::unordered_map<std::string, std::string>& conf_map =got_id_func_cap->second;
        std::unordered_map<std::string, std::string>::const_iterator got_id_func_cap2 = conf_map.find(strFuncname);
        if (got_id_func_cap2 == conf_map.end())
        {
            ANNIWOLOG(INFO) << "not found "<< strFuncname<<" in eventUrl_conf_map,camID:" <<camID;
        }
        else
        {
            submitUrl = got_id_func_cap2->second + suffix;
        }
    }
}


#endif //